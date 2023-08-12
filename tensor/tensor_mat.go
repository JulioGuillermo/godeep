package tensor

import (
	"fmt"
	"strings"

	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tools"
	"github.com/julioguillermo/godeep/types"
)

type TensorMat[T types.Number] struct {
	Shape    []uint
	MulIndex []uint
	Operands []*number.Scalar[T]
	builded  bool
}

func (p *TensorMat[T]) BuildGraph(ctx *context.Context) error {
	if p.builded {
		return nil
	}
	p.builded = true

	for i := range p.Operands {
		ctx.Push(&operation.Scalar[T]{Scalar: p.Operands[i]})
	}
	return nil
}

func (p *TensorMat[_]) GetSize() uint {
	return uint(len(p.Operands))
}

func (p *TensorMat[_]) GetShape() []uint {
	s := make([]uint, len(p.Shape))
	copy(s, p.Shape)
	return s
}

func (p *TensorMat[_]) GetMulIndex() []uint {
	s := make([]uint, len(p.MulIndex))
	copy(s, p.MulIndex)
	return s
}

func (p *TensorMat[T]) GetOperands() []*number.Scalar[T] {
	return p.Operands
}

func (p *TensorMat[T]) GetOperand(index ...uint) (*number.Scalar[T], error) {
	ind, err := tools.GetIndex(p.MulIndex, p.Shape, index)
	if err != nil {
		return nil, err
	}
	if ind >= uint(len(p.Operands)) {
		return nil, errors.FmtNeuralError("Internal index %d out of range %d", ind, len(p.Operands))
	}
	return p.Operands[ind], nil
}

func (p *TensorMat[T]) Bind(t Tensor[T]) error {
	copy(p.Operands, t.GetOperands())
	return nil
}

func (p *TensorMat[T]) LoadFromTensor(t Tensor[T]) error {
	err := tools.GetEqShapeErr("Loading operands from tensor", p.Shape, t.GetShape())
	if err != nil {
		return err
	}
	for i, o := range t.GetOperands() {
		p.Operands[i].Value = o.Value
	}
	return nil
}

func (p *TensorMat[T]) LoadData(data []T) error {
	if uint(len(data)) != p.GetSize() {
		return errors.FmtNeuralError(
			"Invalid data source size %d for a tensor with size %d",
			len(data),
			p.GetSize(),
		)
	}
	for i, o := range p.Operands {
		o.Value = data[i]
	}
	return nil
}

func (p *TensorMat[T]) GetData() []T {
	data := make([]T, p.GetSize())
	for i := range data {
		data[i] = p.Operands[i].Value
	}
	return data
}

func (p *TensorMat[T]) Get(index ...uint) (T, error) {
	o, err := p.GetOperand(index...)
	if err != nil {
		return 0, err
	}
	return o.Value, nil
}

func (p *TensorMat[T]) Set(value T, index ...uint) error {
	o, err := p.GetOperand(index...)
	if err != nil {
		return err
	}
	// o.Set(value)
	o.Value = value
	return nil
}

func (p *TensorMat[_]) String() string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("<Mat (dims: %d, len: %d, shape: ", len(p.Shape), len(p.Operands)))
	for _, s := range p.Shape {
		sb.WriteString(fmt.Sprint(s))
	}
	sb.WriteString(") => [")
	size := len(p.Operands) - 1
	for i, d := range p.Operands {
		sb.WriteString(fmt.Sprint(d.Value))
		if i < size {
			sb.WriteString(", ")
		}
	}
	sb.WriteString("]>")
	return sb.String()
}

func (p *TensorMat[T]) Copy() Tensor[T] {
	ops := make([]*number.Scalar[T], p.GetSize())
	for i := range ops {
		ops[i] = &number.Scalar[T]{
			Value: p.Operands[i].Value,
		}
	}
	return &TensorMat[T]{
		Shape:    p.GetShape(),
		MulIndex: p.GetMulIndex(),
		Operands: ops,
	}
}

func (p *TensorMat[T]) SetBuild(b bool) Tensor[T] {
	p.builded = b
	return p
}
