package tools

import (
	"fmt"
	"strings"

	"github.com/julioguillermo/godeep/errors"
)

func GetIndexMul(shape []uint) []uint {
	if len(shape) == 0 {
		return []uint{}
	}
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

func ReverseIndex(mul, shape []uint, index uint) []uint {
	ind := make([]uint, len(shape))

	for i := range ind {
		ind[i] = index / mul[i]
		index %= mul[i]
	}

	return ind
}

func GetInvertedIndex(index []uint) []uint {
	size := len(index) - 1
	ind := make([]uint, size+1)
	for i := range ind {
		ind[i] = index[size-i]
	}

	return ind
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

func GetEqShapeErr(msg string, s1, s2 []uint) error {
	e := Equals(s1, s2)
	if e == -2 {
		return errors.FmtNeuralError(
			"%s => Shapes with different dimensions %d and %d",
			msg,
			len(s1),
			len(s2),
		)
	}
	if e >= 0 {
		return errors.FmtNeuralError("%s => Different shapes %d and %d at %d", msg, s1[e], s2[e], e)
	}
	return nil
}

func ShapeStr(s []uint) string {
	var sb strings.Builder
	sb.WriteString("[")
	for i, d := range s {
		sb.WriteString(fmt.Sprint(d))
		if i < len(s)-1 {
			sb.WriteString(", ")
		}
	}
	sb.WriteString("]")
	return sb.String()
}
