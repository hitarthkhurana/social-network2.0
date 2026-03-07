import UIKit
import AVFoundation
import CoreVideo

extension UIImage {
    /// Renders the image into a new CVPixelBuffer of the given dimensions.
    /// Used by VideoRecorder to feed frames into AVAssetWriterInputPixelBufferAdaptor.
    func toCVPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width, height,
            kCVPixelFormatType_32BGRA,
            attrs,
            &pixelBuffer)

        guard status == kCVReturnSuccess, let pb = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }

        guard
            let cgImage = self.cgImage,
            let context = CGContext(
                data: CVPixelBufferGetBaseAddress(pb),
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue |
                            CGBitmapInfo.byteOrder32Little.rawValue)
        else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pb
    }
}

extension AVAudioPCMBuffer {
    /// Converts an AVAudioPCMBuffer to a CMSampleBuffer for AVAssetWriterInput.
    func toCMSampleBuffer(presentationTimeStamp: CMTime) -> CMSampleBuffer? {
        var asbd = format.streamDescription.pointee

        var formatDescription: CMAudioFormatDescription?
        guard CMAudioFormatDescriptionCreate(
            allocator: nil,
            asbd: &asbd,
            layoutSize: 0,
            layout: nil,
            magicCookieSize: 0,
            magicCookie: nil,
            extensions: nil,
            formatDescriptionOut: &formatDescription) == noErr,
              let fmt = formatDescription else { return nil }

        let numSamples = CMItemCount(frameLength)
        var timing = CMSampleTimingInfo(
            duration: CMTime(value: 1, timescale: CMTimeScale(format.sampleRate)),
            presentationTimeStamp: presentationTimeStamp,
            decodeTimeStamp: .invalid)

        var sampleBuffer: CMSampleBuffer?
        guard CMSampleBufferCreate(
            allocator: nil,
            dataBuffer: nil,
            dataReady: false,
            makeDataReadyCallback: nil,
            refcon: nil,
            formatDescription: fmt,
            sampleCount: numSamples,
            sampleTimingEntryCount: 1,
            sampleTimingArray: &timing,
            sampleSizeEntryCount: 0,
            sampleSizeArray: nil,
            sampleBufferOut: &sampleBuffer) == noErr,
              let sb = sampleBuffer else { return nil }

        CMSampleBufferSetDataBufferFromAudioBufferList(
            sb,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: 0,
            bufferList: audioBufferList)

        return sb
    }
}
