// Optional offline macOS Vision face-box detector; returns normalized oriented-image coordinates.
import Foundation
import Vision
import ImageIO

struct Face: Encodable { let x: Double; let y: Double; let width: Double; let height: Double; let confidence: Float }
struct Row: Encodable { let path: String; let faces: [Face]; let error: String? }
var rows: [Row] = []
for path in CommandLine.arguments.dropFirst() {
    do {
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw NSError(domain: "image", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not decode image"]) }
        let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let rawOrientation = (props?[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1
        let orientation = CGImagePropertyOrientation(rawValue: rawOrientation) ?? .up
        let request = VNDetectFaceRectanglesRequest()
        request.usesCPUOnly = true
        try VNImageRequestHandler(cgImage: image, orientation: orientation, options: [:]).perform([request])
        let faces = (request.results ?? []).map { observation -> Face in
            let r = observation.boundingBox
            return Face(x: Double(r.minX), y: Double(1-r.maxY), width: Double(r.width), height: Double(r.height), confidence: observation.confidence)
        }
        rows.append(Row(path: path, faces: faces, error: nil))
    } catch {
        rows.append(Row(path: path, faces: [], error: error.localizedDescription))
    }
}
let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
FileHandle.standardOutput.write(try encoder.encode(rows))
