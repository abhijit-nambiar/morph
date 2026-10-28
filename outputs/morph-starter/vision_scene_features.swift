// Local macOS Vision image-feature distances; no semantic labels or segmentation.
import Foundation
import Vision
import ImageIO

struct Result: Encodable { let paths: [String]; let distances: [[Float]]; let revision: Int }
do {
    guard CommandLine.arguments.count == 2 else { throw NSError(domain:"usage",code:1,userInfo:[NSLocalizedDescriptionKey:"Expected a JSON array of image paths"]) }
    let paths = try JSONDecoder().decode([String].self,from: Data(contentsOf: URL(fileURLWithPath:CommandLine.arguments[1])))
    guard !paths.isEmpty && Set(paths).count == paths.count else { throw NSError(domain:"input",code:1,userInfo:[NSLocalizedDescriptionKey:"Paths must be non-empty and unique"]) }
    var observations: [VNFeaturePrintObservation] = []
    for path in paths {
        guard let source=CGImageSourceCreateWithURL(URL(fileURLWithPath:path) as CFURL,nil), let image=CGImageSourceCreateImageAtIndex(source,0,nil) else { throw NSError(domain:"image",code:1,userInfo:[NSLocalizedDescriptionKey:"Cannot decode \(path)"]) }
        let properties=CGImageSourceCopyPropertiesAtIndex(source,0,nil) as? [CFString:Any]
        let orientation=CGImagePropertyOrientation(rawValue:(properties?[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1) ?? .up
        let request=VNGenerateImageFeaturePrintRequest(); request.usesCPUOnly=true
        try VNImageRequestHandler(cgImage:image,orientation:orientation,options:[:]).perform([request])
        guard let observation=request.results?.first as? VNFeaturePrintObservation else { throw NSError(domain:"vision",code:1,userInfo:[NSLocalizedDescriptionKey:"No feature print for \(path)"]) }
        observations.append(observation)
    }
    var matrix=Array(repeating:Array(repeating:Float(0),count:paths.count),count:paths.count)
    for i in 0..<paths.count { for j in 0..<i { var d:Float=0;try observations[i].computeDistance(&d,to:observations[j]);matrix[i][j]=d;matrix[j][i]=d } }
    let result=Result(paths:paths,distances:matrix,revision:VNGenerateImageFeaturePrintRequest.defaultRevision)
    let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys];FileHandle.standardOutput.write(try encoder.encode(result))
} catch { FileHandle.standardError.write(Data(("Vision feature extraction failed: \(error.localizedDescription)\n").utf8));exit(1) }
