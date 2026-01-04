import BinaryParsing
import Foundation
import Testing

@testable import GGUF

@Suite struct TensorInfoTests {
    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, *)
    @Test func `parsing tensor info`() throws {
        var data: [UInt8] = []
        // Name: "weights"
        data += makeGGUFString("weights")
        // Dimension count: 2
        data += [0x02, 0x00, 0x00, 0x00]
        // Dimensions: [1024, 768]
        data += littleEndianBytes(UInt64(1024))
        data += littleEndianBytes(UInt64(768))
        // Data type: F32 (0)
        data += [0x00, 0x00, 0x00, 0x00]
        // Offset: 0
        data += [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]

        var span = ParserSpan(data.span.bytes)
        let info = try GGUF.TensorInfo(parsing: &span)

        #expect(info.name == "weights")
        #expect(info.dimensionCount == 2)
        #expect(info.dimensions == [1024, 768])
        #expect(info.dataType == .f32)
        #expect(info.offset == 0)
        #expect(info.elementCount == 1024 * 768)
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, *)
    @Test func `parsing tensor info with invalid name length should throw error`() throws {
        // Create a name longer than 64 bytes
        let longName = String(repeating: "a", count: 65)
        var data = makeGGUFString(longName)
        // Add minimal tensor info
        data += [0x01, 0x00, 0x00, 0x00]  // 1 dimension
        data += littleEndianBytes(UInt64(100))
        data += [0x00, 0x00, 0x00, 0x00]  // F32
        data += [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]  // offset 0
        var span = ParserSpan(data.span.bytes)

        #expect(throws: GGUF.Error.self) {
            _ = try GGUF.TensorInfo(parsing: &span)
        }
    }
}
