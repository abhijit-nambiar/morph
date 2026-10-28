// Experimental local macOS Vision person mask. Output is a low-resolution grayscale PNG.
import Foundation
import Vision
import ImageIO
import CoreImage
import UniformTypeIdentifiers

do {
    guard CommandLine.arguments.count == 3 else { throw NSError(domain:"usage",code:1,userInfo:[NSLocalizedDescriptionKey:"Expected input image and output PNG"]) }
    let input=URL(fileURLWithPath:CommandLine.arguments[1]);let output=URL(fileURLWithPath:CommandLine.arguments[2])
    guard input.standardizedFileURL != output.standardizedFileURL && !FileManager.default.fileExists(atPath:output.path) else { throw NSError(domain:"output",code:1,userInfo:[NSLocalizedDescriptionKey:"Output exists or would overwrite input"]) }
    guard let source=CGImageSourceCreateWithURL(input as CFURL,nil) else { throw NSError(domain:"image",code:1,userInfo:[NSLocalizedDescriptionKey:"Cannot open input"]) }
    let opts:[CFString:Any]=[kCGImageSourceCreateThumbnailFromImageAlways:true,kCGImageSourceCreateThumbnailWithTransform:true,kCGImageSourceThumbnailMaxPixelSize:1024]
    guard let oriented=CGImageSourceCreateThumbnailAtIndex(source,0,opts as CFDictionary) else { throw NSError(domain:"image",code:2,userInfo:[NSLocalizedDescriptionKey:"Cannot decode input"]) }
    let request=VNGeneratePersonSegmentationRequest();request.qualityLevel = .balanced;request.outputPixelFormat=kCVPixelFormatType_OneComponent8;request.usesCPUOnly=true
    try VNImageRequestHandler(cgImage:oriented,orientation:.up,options:[:]).perform([request])
    guard let result=request.results?.first else { throw NSError(domain:"vision",code:1,userInfo:[NSLocalizedDescriptionKey:"No segmentation mask returned"]) }
    let context=CIContext(options:[.useSoftwareRenderer:true]);let ci=CIImage(cvPixelBuffer:result.pixelBuffer)
    guard let mask=context.createCGImage(ci,from:ci.extent), let dest=CGImageDestinationCreateWithURL(output as CFURL,UTType.png.identifier as CFString,1,nil) else { throw NSError(domain:"output",code:2,userInfo:[NSLocalizedDescriptionKey:"Could not create mask PNG"]) }
    CGImageDestinationAddImage(dest,mask,nil);guard CGImageDestinationFinalize(dest) else { throw NSError(domain:"output",code:3,userInfo:[NSLocalizedDescriptionKey:"Could not write mask PNG"]) }
    print("mask \(mask.width)x\(mask.height); oriented input \(oriented.width)x\(oriented.height)")
} catch { FileHandle.standardError.write(Data(("Segmentation failed: \(error.localizedDescription)\n").utf8));exit(1) }
