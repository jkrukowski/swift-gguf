import BinaryParsing
import Foundation

extension GGUF {
    /// GGUF file header (24 bytes)
    public struct Header: Sendable {
        public let magic: UInt32
        public let version: UInt32
        public let tensorCount: UInt64
        public let metadataKeyValueCount: UInt64

        public init(
            magic: UInt32,
            version: UInt32,
            tensorCount: UInt64,
            metadataKeyValueCount: UInt64
        ) {
            self.magic = magic
            self.version = version
            self.tensorCount = tensorCount
            self.metadataKeyValueCount = metadataKeyValueCount
        }
    }
}

extension GGUF.Header: ExpressibleByParsing {
    public init(parsing input: inout ParserSpan) throws {
        self.magic = try UInt32(parsingBigEndian: &input)
        guard magic == GGUF.Constants.headerMagic else {
            throw GGUF.Error.invalidMagicNumber(magic)
        }
        self.version = try UInt32(parsingLittleEndian: &input)
        guard version == 3 else {
            throw GGUF.Error.notSupportedVersion(version)
        }
        self.tensorCount = try UInt64(parsingLittleEndian: &input)
        self.metadataKeyValueCount = try UInt64(parsingLittleEndian: &input)
    }
}
