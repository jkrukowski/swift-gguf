import BinaryParsing
import Foundation
import Testing

@testable import GGUF

@Suite struct StringTests {
    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, *)
    @Test func `parsing GGUF String`() throws {
        let testString = "test.string.value"
        let data = makeGGUFString(testString)
        var span = ParserSpan(data.span.bytes)
        let parsed = try String(parsingGGUFString: &span)

        #expect(parsed == testString)
        #expect(span.count == 0)
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, *)
    @Test func `parsing empty GGUF String`() throws {
        let data = makeGGUFString("")
        var span = ParserSpan(data.span.bytes)
        let parsed = try String(parsingGGUFString: &span)

        #expect(parsed == "")
        #expect(span.count == 0)
    }
}
