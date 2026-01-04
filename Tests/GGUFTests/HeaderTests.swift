import BinaryParsing
import Foundation
import Testing

@testable import GGUF

@Suite struct HeaderTests {
    @Test func `invalid magic number should throw error`() throws {
        let invalidData: [UInt8] = [
            0x00, 0x00, 0x00, 0x00,  // Wrong magic (should be 0x47 0x47 0x55 0x46)
            0x03, 0x00, 0x00, 0x00,  // Version 3
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // Tensor count: 0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // Metadata count: 0
        ]

        #expect(throws: GGUF.Error.self) {
            _ = try GGUF(parsing: invalidData)
        }
    }

    @Test func `unsupported version should throw error`() throws {
        let invalidData: [UInt8] = [
            0x47, 0x47, 0x55, 0x46,  // Magic "GGUF" (correct big-endian)
            0x02, 0x00, 0x00, 0x00,  // Version 2 (unsupported)
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // Tensor count: 0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // Metadata count: 0
        ]

        #expect(throws: GGUF.Error.self) {
            _ = try GGUF(parsing: invalidData)
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, *)
    @Test func `valid header parsing should succeed`() throws {
        let data = makeGGUFHeader(tensorCount: 5, metadataCount: 3)
        var span = ParserSpan(data.span.bytes)
        let header = try GGUF.Header(parsing: &span)

        #expect(header.magic == GGUF.Constants.headerMagic)
        #expect(header.version == 3)
        #expect(header.tensorCount == 5)
        #expect(header.metadataKeyValueCount == 3)
    }
}
