package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tools"
	"github.com/julioguillermo/godeep/types"
)

type TensorDotProduct[T types.Number] struct {
	TensorMat[T]
	A Tensor[T]
	B Tensor[T]
}

func DotProduct[T types.Number](a, b Tensor[T]) Tensor[T] {
	return &TensorDotProduct[T]{
		A: a,
		B: b,
	}
}

func (p *TensorDotProduct[T]) buildVector(ctx *context.Context) error {
	if p.A.GetSize() != p.B.GetSize() {
		return errors.FmtNeuralError(
			"Dot product fail on vectors with size %d and %d",
			p.A.GetSize(),
			p.B.GetSize(),
		)
	}
	op1 := p.A.GetOperands()
	op2 := p.B.GetOperands()
	ops := make([]*number.Scalar[T], len(op1))
	for i := range ops {
		o := &number.Scalar[T]{}
		ops[i] = o
		ctx.Push(&operation.Mul[T]{
			Scalar: o,
			A:      op1[i],
			B:      op2[i],
		})
	}

	o := &number.Scalar[T]{}
	ctx.Push(&operation.Sum[T]{
		Scalar: o,
		Args:   ops,
	})

	p.Operands = []*number.Scalar[T]{o}
	p.Shape = []uint{1}
	p.MulIndex = []uint{1}

	return nil
}

func (p *TensorDotProduct[T]) buildMatVec(m, v Tensor[T], invert bool, ctx *context.Context) error {
	shape := m.GetShape()
	if invert {
		if shape[len(shape)-2] != v.GetSize() {
			return errors.FmtNeuralError(
				"Invalid dot product on matrix with shape[-2] %d and a vector of size %d",
				shape[len(shape)-2],
				v.GetSize(),
			)
		}
	} else {
		if shape[len(shape)-1] != v.GetSize() {
			return errors.FmtNeuralError(
				"Invalid dot product on matrix with shape[-1] %d and a vector of size %d",
				shape[len(shape)-1],
				v.GetSize(),
			)
		}
	}

	p.Shape = shape[:len(shape)-1]
	if invert {
		p.Shape[len(p.Shape)-1] = shape[len(shape)-1]
	}
	p.MulIndex = tools.GetIndexMul(p.Shape)
	p.Operands = make([]*number.Scalar[T], tools.GetDataSize(p.Shape))

	return p.buildMatVecRecursive(ctx, invert, m, v, 0, 0, []uint{})
}

// TODO better way...
func (p *TensorDotProduct[T]) buildMatVecRecursive(
	ctx *context.Context,
	invert bool,
	m, v Tensor[T],
	dim, index uint,
	oIndex []uint,
) error {
	if dim == uint(len(p.Shape)) {
		ops := make([]*number.Scalar[T], v.GetSize())
		for i, vo := range v.GetOperands() {
			var ind []uint
			if invert {
				ind = make([]uint, len(oIndex)+1)
				copy(ind, oIndex)
				ind[len(ind)-1] = ind[len(ind)-2]
				ind[len(ind)-2] = uint(i)
			} else {
				ind = append(oIndex, uint(i))
			}
			o, e := m.GetOperand(ind...)
			if e != nil {
				return e
			}
			mul := &number.Scalar[T]{}
			ops[i] = mul
			ctx.Push(&operation.Mul[T]{
				Scalar: mul,
				A:      vo,
				B:      o,
			})
		}

		sum := &number.Scalar[T]{}
		p.Operands[index] = sum
		ctx.Push(&operation.Sum[T]{
			Scalar: sum,
			Args:   ops,
		})
		return nil
	}
	for i := uint(0); i < p.Shape[dim]; i++ {
		err := p.buildMatVecRecursive(
			ctx,
			invert,
			m, v,
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

func (p *TensorDotProduct[T]) buildMatMat(ctx *context.Context) error {
	shA := p.A.GetShape()
	shB := p.B.GetShape()
	if shA[len(shA)-1] != shB[len(shB)-2] {
		return errors.FmtNeuralError(
			"Invalid dot product on matrix with shape[%d] %d and matrix with shape[%d] %d",
			len(shA)-1,
			shA[len(shA)-1],
			len(shB)-2,
			shB[len(shB)-2],
		)
	}

	p.Shape = append(shA[:len(shA)-1], shB[:len(shB)-1]...)
	p.MulIndex = tools.GetIndexMul(p.Shape)
	p.Operands = make([]*number.Scalar[T], tools.GetDataSize(p.Shape))
	for i := range p.Operands {
		p.Operands[i] = &number.Scalar[T]{}
	}

	p.buildMatMatRecursive(ctx, uint(len(shA)-1), 0, []uint{}, []uint{}, []uint{})

	return nil
}

func (p *TensorDotProduct[T]) buildMatMatRecursive(
	ctx *context.Context,
	sha uint,
	dim uint,
	index []uint,
	indexA []uint,
	indexB []uint,
) error {
	if dim == uint(len(p.Shape)) {
		ops := make([]*number.Scalar[T], p.A.GetShape()[sha])
		for i := uint(0); i < p.A.GetShape()[sha]; i++ {
			a, err := p.A.GetOperand(append(indexA, i)...)
			if err != nil {
				return nil
			}
			ind := make([]uint, len(indexB)+1)
			copy(ind, indexB)
			ind[len(indexB)] = indexB[len(indexB)-1]
			ind[len(indexB)-1] = i
			b, err := p.B.GetOperand(ind...)
			if err != nil {
				return err
			}
			o := &number.Scalar[T]{}
			ops[i] = o
			ctx.Push(&operation.Mul[T]{
				Scalar: o,
				A:      a,
				B:      b,
			})
		}
		o, err := p.GetOperand(index...)
		if err != nil {
			return err
		}
		ctx.Push(&operation.Sum[T]{
			Scalar: o,
			Args:   ops,
		})
		return nil
	}
	for i := uint(0); i < p.Shape[dim]; i++ {
		if dim < sha {
			err := p.buildMatMatRecursive(
				ctx,
				sha,
				dim+1,
				append(index, i),
				append(indexA, i),
				indexB,
			)
			if err != nil {
				return err
			}
		} else {
			err := p.buildMatMatRecursive(ctx, sha, dim+1, append(index, i), indexA, append(indexB, i))
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (p *TensorDotProduct[T]) BuildGraph(ctx *context.Context) error {
	if p.builded {
		return nil
	}
	p.builded = true

	err := p.A.BuildGraph(ctx)
	if err != nil {
		return err
	}
	err = p.B.BuildGraph(ctx)
	if err != nil {
		return err
	}

	if len(p.A.GetShape()) == 1 && len(p.B.GetShape()) == 1 {
		return p.buildVector(ctx)
	}

	if len(p.B.GetShape()) == 1 {
		return p.buildMatVec(p.A, p.B, false, ctx)
	}

	if len(p.A.GetShape()) == 1 {
		return p.buildMatVec(p.B, p.A, true, ctx)
	}

	// if len(p.A.GetShape()) == 2 && len(p.B.GetShape()) == 2 {
	return p.buildMatMat(ctx)
	//}

	//return errors.FmtNeuralError(
	//	"Invalid dot operation on shapes %s and %s",
	//	tools.ShapeStr(p.A.GetShape()),
	//	tools.ShapeStr(p.B.GetShape()),
	//)
}
