package layer

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

type SoftMax[T types.Number] struct {
	Base[T]
	Min *number.Scalar[T]
	Sum *number.Scalar[T]
}

func NewSoftMax[T types.Number]() Layer[T] {
	l := &SoftMax[T]{}
	l.Type = "SoftMax"
	return l
}

func (p *SoftMax[T]) Build() (uint, error) {
	if p.CheckB() {
		return p.Index, nil
	}

	err := p.PreBuild()
	if err != nil {
		return 0, err
	}

	if p.PreLayer == nil {
		return 0, p.Error("This layer can not be input layer")
	}
	// p.Activation = p.PreLayer.GetActivation()
	return p.Index, nil
}

func (p *SoftMax[T]) BuildFeedforward(ctx *context.Context) error {
	if p.CheckFF() {
		return nil
	}
	err := p.PreBuildFeedforward(ctx)
	if err != nil {
		return err
	}

	//p.Min = &number.Scalar[T]{}
	//ctx.Push(&operation.Min[T]{
	//	Scalar: p.Min,
	//	Args:   p.Input.GetOperands(),
	//})

	// t := tensor.AddScalar[T](p.Input, p.Min)
	// t.BuildGraph(ctx)

	//p.Sum = &number.Scalar[T]{}
	//ctx.Push(&operation.Sum[T]{
	//	Scalar: p.Sum,
	//	Args:   t.GetOperands(),
	//})

	// p.Neta = tensor.DivScalar[T](t, p.Sum)
	p.Output = tensor.SoftMax[T](p.Input)
	// p.Output = p.Neta // tensor.NewZeros[T](outShape...)

	err = p.Output.BuildGraph(ctx)
	if err != nil {
		return err
	}

	p.Dif = p.PreLayer.GetDif()

	return nil
}

func (p *SoftMax[T]) BuildBackpropagation(ctx *context.Context, a, m *number.Scalar[T]) error {
	if p.CheckBP() {
		return nil
	}

	Dif := p.Dif
	if p.Ref.Value > 1 {
		Dif = tensor.DivScalar(Dif, p.Ref)
	}
	err := Dif.BuildGraph(ctx)
	if err != nil {
		return err
	}

	return p.PostBuildBackpropagation(ctx, a, m)
}

func (p *SoftMax[T]) BuildDer(ctx *context.Context) (tensor.Tensor[T], error) {
	if p.Der == nil {
		der, err := p.PreLayer.BuildDer(ctx)
		if err != nil {
			return nil, err
		}
		p.Der = der
	}
	return p.Der, nil
}
