import Foundation
import GGUF
import Numerics
import Testing

/// Creates a GGUF header with the specified counts
func makeGGUFHeader(
    tensorCount: UInt64 = 0,
    metadataCount: UInt64 = 0
) -> [UInt8] {
    var data: [UInt8] = []
    data += [0x47, 0x47, 0x55, 0x46]  // Magic "GGUF" (big-endian bytes: G G U F)
    data += [0x03, 0x00, 0x00, 0x00]  // Version 3 (little-endian)
    data += littleEndianBytes(tensorCount)
    data += littleEndianBytes(metadataCount)
    return data
}

/// Creates a GGUF string (length-prefixed UTF-8)
func makeGGUFString(_ string: String) -> [UInt8] {
    var data = littleEndianBytes(UInt64(string.utf8.count))
    data += Array(string.utf8)
    return data
}

/// Converts an integer to little-endian byte array
func littleEndianBytes<T: FixedWidthInteger>(_ value: T) -> [UInt8] {
    Swift.withUnsafeBytes(of: value.littleEndian) { Array($0) }
}

func allClose<T: Numeric>(
    _ lhs: [T],
    _ rhs: [T],
    absoluteTolerance: T.Magnitude = T.Magnitude.ulpOfOne.squareRoot()
        * T.Magnitude.leastNormalMagnitude,
    relativeTolerance: T.Magnitude = T.Magnitude.ulpOfOne.squareRoot()
) -> Bool where T.Magnitude: FloatingPoint {
    guard lhs.count == rhs.count else {
        Issue.record("Sizes differ: \(lhs.count) vs. \(rhs.count)")
        return false
    }
    for (l, r) in zip(lhs, rhs) {
        guard
            l.isApproximatelyEqual(
                to: r,
                absoluteTolerance: absoluteTolerance,
                relativeTolerance: relativeTolerance
            )
        else {
            Issue.record("Expected \(lhs) to be approximately equal to \(rhs), but \(l) != \(r)")
            return false
        }
    }
    return true
}

func isClose(_ lhs: GGUF.MetadataValue, _ rhs: GGUF.MetadataValue) -> Bool {
    switch (lhs, rhs) {
    case (.uint8(let lvalue), .uint8(let rvalue)):
        return lvalue == rvalue
    case (.uint16(let lvalue), .uint16(let rvalue)):
        return lvalue == rvalue
    case (.uint32(let lvalue), .uint32(let rvalue)):
        return lvalue == rvalue
    case (.uint64(let lvalue), .uint64(let rvalue)):
        return lvalue == rvalue
    case (.int8(let lvalue), .int8(let rvalue)):
        return lvalue == rvalue
    case (.int16(let lvalue), .int16(let rvalue)):
        return lvalue == rvalue
    case (.int32(let lvalue), .int32(let rvalue)):
        return lvalue == rvalue
    case (.int64(let lvalue), .int64(let rvalue)):
        return lvalue == rvalue
    case (.float32(let lvalue), .float32(let rvalue)):
        return lvalue.isApproximatelyEqual(to: rvalue)
    case (.float64(let lvalue), .float64(let rvalue)):
        return lvalue.isApproximatelyEqual(to: rvalue)
    case (.string(let lvalue), .string(let rvalue)):
        return lvalue == rvalue
    case (.bool(let lvalue), .bool(let rvalue)):
        return lvalue == rvalue
    case (.array(let lvalue, let lmetadata), .array(let rvalue, let rmetadata)):
        return lvalue == rvalue && lmetadata == rmetadata
    default:
        return false
    }
}
