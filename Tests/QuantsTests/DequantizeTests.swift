import Foundation
import Quants
import TestData
import Testing

func deqantizeFn(byName name: String) -> (Data, Int) -> [Float] {
    switch name {
    case "Q2_K":
        return Dequantize.Q2_K
    case "Q3_K":
        return Dequantize.Q3_K
    case "Q4_0":
        return Dequantize.Q4_0
    case "Q4_1":
        return Dequantize.Q4_1
    case "Q4_K":
        return Dequantize.Q4_K
    case "Q5_0":
        return Dequantize.Q5_0
    case "Q5_1":
        return Dequantize.Q5_1
    case "Q5_K":
        return Dequantize.Q5_K
    case "Q6_K":
        return Dequantize.Q6_K
    case "Q8_0":
        return Dequantize.Q8_0
    case "Q8_K":
        return Dequantize.Q8_K
    case "IQ4_NL":
        return Dequantize.IQ4_NL
    default:
        fatalError("Unsupported quantized type: \(name)")
    }
}

@Test(arguments: quantizedValuesByName)
func `quantized tensor data should dequantize`(_ pair: (name: String, values: [Float])) throws {
    let tensorData = try #require(testData(named: pair.name, withExtension: "bin"))
    let deqantize = deqantizeFn(byName: pair.name)
    let tensorCount = 2048 * 256
    let tensor = deqantize(tensorData, tensorCount)

    #expect(tensor.count == tensorCount)
    #expect(
        allClose(
            Array(tensor[..<32]),
            pair.values
        )
    )
}
