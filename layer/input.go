package layer

import (
	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

type Input[T types.Number] struct {
	Base[T]
}

func NewInput[T types.Number](shape ...uint) Layer[T] {
	in := &Input[T]{
		Base: Base[T]{
			Input:      tensor.NewZeros[T](shape...),
			Dif:        tensor.NewZeros[T](shape...),
			Activation: activation.Linear[T]{},
		},
	}
	return in
}

func (p *Input[T]) Build() (uint, error) {
	if p.CheckB() {
		return p.Index, nil
	}

	err := p.PreBuild()
	if err != nil {
		return 0, err
	}

	if p.PreLayer != nil {
		p.PreLayer.GetRef().Value += p.Ref.Value
		p.Ref = p.PreLayer.GetRef()
		// p.Activation = p.PreLayer.GetActivation()
	}

	return p.Index, nil
}

func (p *Input[T]) BuildFeedforward(ctx *context.Context) error {
	if p.CheckFF() {
		return nil
	}
	//err := p.PreBuildFeedforward(ctx)
	//if err != nil {
	//	return err
	//}

	if p.PreLayer != nil {
		return p.PreLayer.BuildFeedforward(ctx)
		//p.Input = p.PreLayer.GetOutputs()
		//p.Output = p.PreLayer.GetOutputs()
		//// p.Neta = p.PreLayer.GetNetas()
		//p.Ref = p.PreLayer.GetRef()
		//// p.Activation = p.PreLayer.GetActivation()
		//return nil
	}
	p.Neta = p.Input
	p.Output = p.Input
	p.Dif = tensor.NewZeros[T](p.Output.GetShape()...)
	return nil
}

func (p *Input[T]) BuildBackpropagation(
	ctx *context.Context,
	a *number.Scalar[T],
	m *number.Scalar[T],
) error {
	if p.CheckBP() {
		return nil
	}
	//if p.PreLayer != nil {
	//	err := tensor.Transfer[T](ctx, p.Dif, p.PreLayer.GetDif())
	//	if err != nil {
	//		return err
	//	}
	//}
	//return p.PostBuildBackpropagation(ctx, a, m)
	if p.PreLayer != nil {
		return p.PreLayer.BuildBackpropagation(ctx, a, m)
	}
	return nil
}

func (p *Input[T]) GetInputs() tensor.Tensor[T] {
	if p.PreLayer != nil {
		return p.PreLayer.GetOutputs()
	}
	return p.Input
}

func (p *Input[T]) GetOutputs() tensor.Tensor[T] {
	if p.PreLayer != nil {
		return p.PreLayer.GetOutputs()
	}
	return p.Output
}

func (p *Input[T]) GetDif() tensor.Tensor[T] {
	if p.PreLayer != nil {
		return p.PreLayer.GetDif()
	}
	return p.Dif
}

func (p *Input[T]) BuildDer(ctx *context.Context) (tensor.Tensor[T], error) {
	if p.PreLayer != nil {
		return p.PreLayer.BuildDer(ctx)
	}
	if p.Der == nil {
		p.Der = tensor.Activate[T](p.Neta, p.Activation.Derive)
		err := p.Der.BuildGraph(ctx)
		if err != nil {
			return nil, err
		}
	}
	return p.Der, nil
}
