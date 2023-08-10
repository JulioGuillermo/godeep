package tools

import "github.com/julioguillermo/neuralnetwork/v2/errors"

func GetIndexMul(shape []uint) []uint {
	size := len(shape)
	mulIndex := make([]uint, size)

	mul := uint(1)
	mulIndex[size-1] = mul
	for i := size - 1; i > 0; i-- {
		mul *= shape[i]
		mulIndex[i-1] = mul
	}

	return mulIndex
}

func GetDataSize(shape []uint) uint {
	size := len(shape)
	mul := uint(1)
	for i := size - 1; i >= 0; i-- {
		mul *= shape[i]
	}
	return mul
}

func GetIndex(mul, shape, index []uint) (uint, error) {
	if len(mul) != len(shape) {
		return 0, errors.FmtNeuralError(
			"Invalid internal dimensions: %d == %d",
			len(shape),
			len(mul),
		)
	}
	if len(shape) != len(index) {
		return 0, errors.FmtNeuralError(
			"Invalid index dimensions %d for internal dimensions %d",
			len(index),
			len(shape),
		)
	}

	for i, s := range shape {
		if index[i] >= s {
			return 0, errors.FmtNeuralError("Index %d at %d out of range %d", index[i], i, s)
		}
	}

	ind := uint(0)
	for i, in := range index {
		ind += in * mul[i]
	}

	return ind, nil
}

func Equals(shape1, shape2 []uint) int {
	if len(shape1) != len(shape2) {
		return -2
	}
	for i, s := range shape1 {
		if shape2[i] != s {
			return i
		}
	}
	return -1
}

func GetEqShapeErr(s1, s2 []uint) error {
	e := Equals(s1, s2)
	if e == -2 {
		return errors.FmtNeuralError("Shapes with different dimensions %d and %d", len(s1), len(s2))
	}
	if e >= 0 {
		return errors.FmtNeuralError("Different shapes %d and %d at %d", s1[e], s2[e], e)
	}
	return nil
}
