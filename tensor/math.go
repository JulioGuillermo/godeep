package tensor

import (
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/types"
)

func Add[T types.Number](a, b Tensor[T]) Tensor[T] { return NewOp2[T](a, b, Op2Add) }
func Sub[T types.Number](a, b Tensor[T]) Tensor[T] { return NewOp2[T](a, b, Op2Sub) }
func Mul[T types.Number](a, b Tensor[T]) Tensor[T] { return NewOp2[T](a, b, Op2Mul) }
func Div[T types.Number](a, b Tensor[T]) Tensor[T] { return NewOp2[T](a, b, Op2Div) }

func AddScalar[T types.Number](t Tensor[T], s *number.Scalar[T]) Tensor[T] {
	return NewOpScalar[T](t, s, OpSAdd)
}

func SubScalar[T types.Number](t Tensor[T], s *number.Scalar[T]) Tensor[T] {
	return NewOpScalar[T](t, s, OpSSub)
}

func MulScalar[T types.Number](t Tensor[T], s *number.Scalar[T]) Tensor[T] {
	return NewOpScalar[T](t, s, OpSMul)
}

func DivScalar[T types.Number](t Tensor[T], s *number.Scalar[T]) Tensor[T] {
	return NewOpScalar[T](t, s, OpSDiv)
}

func Sum[T types.Number](t Tensor[T]) Tensor[T] { return TensorMath[T](t, TensorSum) }
func Avg[T types.Number](t Tensor[T]) Tensor[T] { return TensorMath[T](t, TensorAvg) }
func Max[T types.Number](t Tensor[T]) Tensor[T] { return TensorMath[T](t, TensorMax) }
func Min[T types.Number](t Tensor[T]) Tensor[T] { return TensorMath[T](t, TensorMin) }

func DSum[T types.Number](t Tensor[T], d uint) Tensor[T] { return DimMath[T](t, d, DimSum) }
func DAvg[T types.Number](t Tensor[T], d uint) Tensor[T] { return DimMath[T](t, d, DimAvg) }
func DMax[T types.Number](t Tensor[T], d uint) Tensor[T] { return DimMath[T](t, d, DimMax) }
func DMin[T types.Number](t Tensor[T], d uint) Tensor[T] { return DimMath[T](t, d, DimMin) }
