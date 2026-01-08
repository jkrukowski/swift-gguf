import Foundation
import TestData
import Testing

@testable import GGUF

@Test(arguments: ["F32", "F64", "F16", "I8", "I16", "I32", "I64"])
func `float array should be present for different tensor data types`(_ resource: String) throws {
    let fileData = try #require(
        testData(named: resource, withExtension: "gguf")
    )
    let gguf = try GGUF(parsing: fileData)
    let floatArray = try gguf.tensorFloatArray(at: 0, from: fileData)

    #expect(floatArray.count == 32)
    #expect(
        allClose(
            floatArray,
            [
                1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0,
                9.0, -10.0, 11.0, -12.0, 13.0, -14.0, 15.0, -16.0,
                17.0, -18.0, 19.0, -20.0, 21.0, -22.0, 23.0, -24.0,
                25.0, -26.0, 27.0, -28.0, 29.0, -30.0, 31.0, -32.0,
            ]
        )
    )
}
