// Offline macOS Vision detector with overlapping analysis regions. Does not modify photos.
import Foundation
import Vision
import ImageIO

struct Face: Encodable { let x: Double; let y: Double; let width: Double; let height: Double; let confidence: Float }
struct Row: Encodable { let path: String; let faces: [Face]; let error: String? }
func overlap(_ a: Face, _ b: Face) -> Double {
    let intersection=max(0,min(a.x+a.width,b.x+b.width)-max(a.x,b.x))*max(0,min(a.y+a.height,b.y+b.height)-max(a.y,b.y))
    let union=a.width*a.height+b.width*b.height-intersection
    return union > 0 ? intersection/union : 0
}
var rows:[Row]=[]
for path in CommandLine.arguments.dropFirst() {
    do {
        guard let source=CGImageSourceCreateWithURL(URL(fileURLWithPath:path) as CFURL,nil) else { throw NSError(domain:"image",code:1,userInfo:[NSLocalizedDescriptionKey:"Could not open image"]) }
        let options:[CFString:Any]=[kCGImageSourceCreateThumbnailFromImageAlways:true,kCGImageSourceCreateThumbnailWithTransform:true,kCGImageSourceThumbnailMaxPixelSize:6048]
        guard let image=CGImageSourceCreateThumbnailAtIndex(source,0,options as CFDictionary) else { throw NSError(domain:"image",code:2,userInfo:[NSLocalizedDescriptionKey:"Could not decode image"]) }
        let width=image.width,height=image.height;var candidates:[Face]=[]
        for row in 0...2 { for col in 0...2 {
            let x0=Int(Double(width)*Double(col)*0.3);let y0=Int(Double(height)*Double(row)*0.3)
            let x1=min(width,Int(Double(width)*(Double(col)*0.3+0.4)));let y1=min(height,Int(Double(height)*(Double(row)*0.3+0.4)))
            guard let tile=image.cropping(to:CGRect(x:x0,y:y0,width:x1-x0,height:y1-y0)) else { throw NSError(domain:"image",code:3,userInfo:[NSLocalizedDescriptionKey:"Could not create analysis region"]) }
            let request=VNDetectFaceRectanglesRequest();request.usesCPUOnly=true
            try VNImageRequestHandler(cgImage:tile,orientation:.up,options:[:]).perform([request])
            for obs in request.results ?? [] {
                let r=obs.boundingBox;let tw=Double(x1-x0),th=Double(y1-y0)
                candidates.append(Face(x:(Double(x0)+r.minX*tw)/Double(width),y:(Double(y0)+(1-r.maxY)*th)/Double(height),width:r.width*tw/Double(width),height:r.height*th/Double(height),confidence:obs.confidence))
            }
        }}
        var faces:[Face]=[]
        for f in candidates.sorted(by:{$0.confidence > $1.confidence}) where !faces.contains(where:{overlap($0,f)>0.35}) { faces.append(f) }
        rows.append(Row(path:path,faces:faces,error:nil))
    } catch { rows.append(Row(path:path,faces:[],error:error.localizedDescription)) }
}
let encoder=JSONEncoder();encoder.outputFormatting=[.prettyPrinted,.sortedKeys];FileHandle.standardOutput.write(try encoder.encode(rows))
