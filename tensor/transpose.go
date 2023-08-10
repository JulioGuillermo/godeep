package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tools"
	"github.com/julioguillermo/godeep/types"
)

type TensorTanspose[T types.Number] struct {
	TensorMat[T]
	T Tensor[T]
}

func Transpose[T types.Number](t Tensor[T]) Tensor[T] {
	return &TensorTanspose[T]{
		T: t,
	}
}

func (p *TensorTanspose[T]) BuildGraph(ctx *context.Context) error {
	if p.builded {
		return nil
	}
	p.builded = true

	err := p.T.BuildGraph(ctx)
	if err != nil {
		return err
	}
	TShape := p.T.GetShape()

	dims := len(TShape)
	p.Shape = make([]uint, dims)
	for i := range p.Shape {
		p.Shape[i] = TShape[dims-i-1]
	}
	p.MulIndex = tools.GetIndexMul(p.Shape)

	p.Operands = make([]*operation.Operand[T], p.T.GetSize())

	return p.transposeRecursive(0, 0, []uint{})
}

// Función auxiliar recursiva para calcular la traspuesta de la matriz
// TODO better way...
func (p *TensorTanspose[_]) transposeRecursive(dim, index uint, oIndex []uint) error {
	if dim == uint(len(p.Shape)) {
		size := len(oIndex)
		rIndex := make([]uint, size)
		for i, ii := range oIndex {
			rIndex[size-i-1] = ii
		}
		o, err := p.T.GetOperand(rIndex...)
		if err != nil {
			return err
		}
		p.Operands[index] = o
		return nil
	}
	for i := uint(0); i < p.Shape[dim]; i++ {
		err := p.transposeRecursive(
			dim+1,
			index+i*p.MulIndex[dim],
			append(oIndex, i),
		)
		if err != nil {
			return err
		}
	}
	return nil
}
