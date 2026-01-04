import BinaryParsing
import Foundation
import Testing

@testable import GGUF

@Suite struct MetadataTests {
    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, *)
    @Test func `parsing uint8`() throws {
        let data: [UInt8] = [42]
        var span = ParserSpan(data.span.bytes)
        let value = try GGUF.MetadataValue(parsing: &span, type: .uint8)

        #expect(isClose(value, .uint8(42)))
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, *)
    @Test func `parsing int32`() throws {
        let data = littleEndianBytes(Int32(-12345))
        var span = ParserSpan(data.span.bytes)
        let value = try GGUF.MetadataValue(parsing: &span, type: .int32)

        #expect(isClose(value, .int32(-12345)))
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, *)
    @Test func `parsing float32`() throws {
        let floatValue: Float = 3.14159
        let data = littleEndianBytes(floatValue.bitPattern)
        var span = ParserSpan(data.span.bytes)
        let value = try GGUF.MetadataValue(parsing: &span, type: .float32)

        #expect(isClose(value, .float32(3.14159)))
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, *)
    @Test func `parsing bool true`() throws {
        let data: [UInt8] = [1]
        var span = ParserSpan(data.span.bytes)
        let value = try GGUF.MetadataValue(parsing: &span, type: .bool)

        #expect(isClose(value, .bool(true)))
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, *)
    @Test func `parsing bool false`() throws {
        let data: [UInt8] = [0]
        var span = ParserSpan(data.span.bytes)
        let value = try GGUF.MetadataValue(parsing: &span, type: .bool)

        #expect(isClose(value, .bool(false)))
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, *)
    @Test func `parsing string`() throws {
        let testString = "model.name"
        let data = makeGGUFString(testString)
        var span = ParserSpan(data.span.bytes)
        let value = try GGUF.MetadataValue(parsing: &span, type: .string)

        #expect(isClose(value, .string("model.name")))
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, *)
    @Test func `parsing uint64`() throws {
        let data = littleEndianBytes(UInt64(0x1234_5678_9ABC_DEF0))
        var span = ParserSpan(data.span.bytes)
        let value = try GGUF.MetadataValue(parsing: &span, type: .uint64)

        #expect(isClose(value, .uint64(0x1234_5678_9ABC_DEF0)))
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, *)
    @Test func `parsing float64`() throws {
        let doubleValue: Double = 2.718281828459045
        let data = littleEndianBytes(doubleValue.bitPattern)
        var span = ParserSpan(data.span.bytes)
        let value = try GGUF.MetadataValue(parsing: &span, type: .float64)

        #expect(isClose(value, .float64(2.718281828459045)))
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, *)
    @Test func `parsing array`() throws {
        var data: [UInt8] = []
        // Array type (uint32 = 4)
        data += [0x04, 0x00, 0x00, 0x00]
        // Array length (3)
        data += [0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        // Three uint32 values: 10, 20, 30
        data += [0x0A, 0x00, 0x00, 0x00]
        data += [0x14, 0x00, 0x00, 0x00]
        data += [0x1E, 0x00, 0x00, 0x00]

        var span = ParserSpan(data.span.bytes)
        let value = try GGUF.MetadataValue(parsing: &span, type: .array)

        #expect(value == .array(.uint32, [.uint32(10), .uint32(20), .uint32(30)]))
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, *)
    @Test func `parsing key value`() throws {
        var data: [UInt8] = []
        // Key: "general.alignment"
        data += makeGGUFString("general.alignment")
        // Value type: uint32 (4)
        data += [0x04, 0x00, 0x00, 0x00]
        // Value: 32
        data += [0x20, 0x00, 0x00, 0x00]

        var span = ParserSpan(data.span.bytes)
        let kv = try GGUF.MetadataKeyValue(parsing: &span)

        #expect(kv.key == "general.alignment")
        #expect(kv.valueType == .uint32)
        #expect(kv.value == .uint32(32))
    }
}
