package tensor

import (
	"math/rand"

	"github.com/julioguillermo/neuralnetwork/v2/operation"
	"github.com/julioguillermo/neuralnetwork/v2/tools"
	"github.com/julioguillermo/neuralnetwork/v2/types"
)

func initMat[T types.Number](shape []uint) *TensorMat[T] {
	return &TensorMat[T]{
		Shape:    shape,
		MulIndex: tools.GetIndexMul(shape),
	}
}

func NewZeros[T types.Number](shape ...uint) Tensor[T] {
	data := make([]*operation.Operand[T], tools.GetDataSize(shape))
	for i := range data {
		data[i] = &operation.Operand[T]{Value: 0}
	}
	mat := initMat[T](shape)
	mat.Operands = data
	return mat
}

func NewOne[T types.Number](shape ...uint) Tensor[T] {
	data := make([]*operation.Operand[T], tools.GetDataSize(shape))
	for i := range data {
		data[i] = &operation.Operand[T]{Value: 1}
	}
	mat := initMat[T](shape)
	mat.Operands = data
	return mat
}

func NewRand[T types.Number](min, max float64, shape ...uint) Tensor[T] {
	data := make([]*operation.Operand[T], tools.GetDataSize(shape))
	for i := range data {
		data[i] = &operation.Operand[T]{Value: T(rand.Float64()*(max-min) + min)}
	}
	mat := initMat[T](shape)
	mat.Operands = data
	return mat
}

func NewNormRand[T types.Number](shape ...uint) Tensor[T] {
	data := make([]*operation.Operand[T], tools.GetDataSize(shape))
	for i := range data {
		data[i] = &operation.Operand[T]{Value: T(rand.NormFloat64())}
	}
	mat := initMat[T](shape)
	mat.Operands = data
	return mat
}

func NewFromValues[T types.Number](values []T, shape ...uint) Tensor[T] {
	data := make([]*operation.Operand[T], tools.GetDataSize(shape))

	size := len(data)
	total := len(data)
	if len(values) < size {
		size = len(values)
	}

	for i := 0; i < size; i++ {
		data[i] = &operation.Operand[T]{Value: values[i]}
	}
	for i := size; i < total; i++ {
		data[i] = &operation.Operand[T]{Value: 0}
	}

	mat := initMat[T](shape)
	mat.Operands = data
	return mat
}
