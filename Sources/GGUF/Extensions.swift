import BinaryParsing

extension String {
    /// Parses a GGUF string: UInt64 length + UTF-8 data (no null terminator)
    public init(parsingGGUFString input: inout ParserSpan) throws {
        let length = try UInt64(parsingLittleEndian: &input)
        guard let lengthInt = Int(exactly: length) else {
            throw GGUF.Error.invalidStringCount(length)
        }
        self = try String(parsingUTF8: &input, count: lengthInt)
    }
}
