package layer

import (
	"github.com/julioguillermo/neuralnetwork/v2/activation"
	"github.com/julioguillermo/neuralnetwork/v2/context"
	"github.com/julioguillermo/neuralnetwork/v2/errors"
	"github.com/julioguillermo/neuralnetwork/v2/operation"
	"github.com/julioguillermo/neuralnetwork/v2/tensor"
	"github.com/julioguillermo/neuralnetwork/v2/types"
)

type Dense[T types.Number] struct {
	Weights tensor.Tensor[T]
	Bias    tensor.Tensor[T]

	NWeights tensor.Tensor[T]
	MWeights tensor.Tensor[T]
	NBias    tensor.Tensor[T]
	MBias    tensor.Tensor[T]
	Alpha    *operation.Operand[T]
	Momentum *operation.Operand[T]

	PreLayer Layer[T]

	Input tensor.Tensor[T]

	Neta   tensor.Tensor[T]
	Output tensor.Tensor[T]
	Dif    tensor.Tensor[T]

	Activation activation.Activation[T]

	Trainable bool
}

func NewDense[T types.Number](outs uint, act activation.Activation[T]) Layer[T] {
	outputs := tensor.NewZeros[T](outs)
	return &Dense[T]{
		Output:     outputs,
		Activation: act,
		Trainable:  true,
	}
}

func NewInDense[T types.Number](ins, outs uint, act activation.Activation[T]) Layer[T] {
	inputs := tensor.NewZeros[T](ins)
	outputs := tensor.NewZeros[T](outs)
	return &Dense[T]{
		Input:      inputs,
		Output:     outputs,
		Activation: act,
		Trainable:  true,
	}
}

func (p *Dense[T]) GetInputs() tensor.Tensor[T] {
	return p.Input
}

func (p *Dense[T]) GetOutputs() tensor.Tensor[T] {
	return p.Output
}

func (p *Dense[T]) GetNetas() tensor.Tensor[T] {
	return p.Neta
}

func (p *Dense[T]) GetDif() tensor.Tensor[T] {
	return p.Dif
}

func (p *Dense[T]) GetActivation() activation.Activation[T] {
	return p.Activation
}

func (p *Dense[T]) Conect(prelayer Layer[T]) {
	p.PreLayer = prelayer
}

func (p *Dense[T]) Build() error {
	if p.PreLayer != nil {
		err := p.PreLayer.Build()
		if err != nil {
			return err
		}
		p.Input = p.PreLayer.GetOutputs()
	}
	if p.Input == nil {
		return errors.FmtNeuralError("Invalid layer input => nil")
	}

	inputs := p.Input.GetSize()
	outputs := p.Output.GetSize()

	p.Neta = tensor.NewZeros[T](p.Output.GetShape()...)
	p.Dif = tensor.NewZeros[T](p.Output.GetShape()...)

	p.Bias = tensor.NewNormRand[T](outputs)
	p.Weights = tensor.NewNormRand[T](outputs, inputs)

	return nil
}

func (p *Dense[T]) BuildFeedforward(ctx *context.Context) error {
	if p.PreLayer != nil {
		err := p.PreLayer.BuildFeedforward(ctx)
		if err != nil {
			return err
		}
	}

	inputs := p.Input.GetSize()
	outputs := p.Output.GetSize()

	for i := uint(0); i < outputs; i++ {
		ops := make([]*operation.Operand[T], inputs+1)
		bias, err := p.Bias.GetOperand(i)
		if err != nil {
			return err
		}
		ops[inputs] = bias

		for j := uint(0); j < inputs; j++ {
			mul_iw := &operation.Operand[T]{}
			ops[j] = mul_iw
			weight, err := p.Weights.GetOperand(i, j)
			if err != nil {
				return err
			}
			input := p.Input.GetOperands()[j]
			ctx.Push(&operation.Mul[T]{
				Operand: mul_iw,
				A:       input,
				B:       weight,
			})
		}

		neta, err := p.Neta.GetOperand(i)
		if err != nil {
			return err
		}
		ctx.Push(&operation.Sum[T]{
			Operand: neta,
			Args:    ops,
		})

		output, err := p.Output.GetOperand(i)
		if err != nil {
			return err
		}
		ctx.Push(&operation.Func[T]{
			Operand: output,
			O:       neta,
			F:       p.Activation.Activate,
		})
	}

	return nil
}

func (p *Dense[T]) BuildBackpropagation(
	ctx *context.Context,
	Alpha, Momentum *operation.Operand[T],
) error {
	inputs := p.Input.GetSize()
	outputs := p.Output.GetSize()

	p.NWeights = tensor.NewZeros[T](p.Weights.GetShape()...)
	p.MWeights = tensor.NewZeros[T](p.Weights.GetShape()...)

	p.NBias = tensor.NewZeros[T](p.Bias.GetShape()...)
	p.MBias = tensor.NewZeros[T](p.Bias.GetShape()...)

	if p.PreLayer != nil {
		der := tensor.Activate(p.PreLayer.GetNetas(), p.PreLayer.GetActivation().Derive)
		err := der.BuildGraph(ctx)
		if err != nil {
			return err
		}

		for i := uint(0); i < inputs; i++ {
			ops := make([]*operation.Operand[T], outputs)
			for j := uint(0); j < outputs; j++ {
				dif, err := p.Dif.GetOperand(j)
				if err != nil {
					return err
				}
				w, err := p.Weights.GetOperand(j, i)
				if err != nil {
					return err
				}
				o := &operation.Operand[T]{}
				ctx.Push(&operation.Mul[T]{
					Operand: o,
					A:       dif,
					B:       w,
				})
				ops[j] = o
			}
			sum := &operation.Operand[T]{}
			ctx.Push(&operation.Sum[T]{
				Operand: sum,
				Args:    ops,
			})
			pd := p.PreLayer.GetDif().GetOperands()[i]
			d := der.GetOperands()[i]
			ctx.Push(&operation.Mul[T]{
				Operand: pd,
				A:       sum,
				B:       d,
			})
		}
	}

	for i := uint(0); i < outputs; i++ {
		dif, err := p.Dif.GetOperand(i)
		if err != nil {
			return err
		}
		mul_dif_alpha := &operation.Operand[T]{}
		ctx.Push(&operation.Mul[T]{
			Operand: mul_dif_alpha,
			A:       dif,
			B:       Alpha,
		})
		momentum, err := p.MBias.GetOperand(i)
		if err != nil {
			return err
		}
		moment := &operation.Operand[T]{}
		ctx.Push(&operation.Mul[T]{
			Operand: moment,
			A:       momentum,
			B:       Momentum,
		})
		bias, err := p.Bias.GetOperand(i)
		if err != nil {
			return err
		}
		new_bias, err := p.NBias.GetOperand(i)
		if err != nil {
			return err
		}
		ctx.Push(&operation.Sum[T]{
			Operand: new_bias,
			Args:    []*operation.Operand[T]{bias, moment, mul_dif_alpha},
		})

		for j := uint(0); j < inputs; j++ {
			input := p.Input.GetOperands()[j]
			momentum, err := p.MWeights.GetOperand(i, j)
			if err != nil {
				return err
			}
			delta := &operation.Operand[T]{}
			ctx.Push(&operation.Mul[T]{
				Operand: delta,
				A:       mul_dif_alpha,
				B:       input,
			})
			moment := &operation.Operand[T]{}
			ctx.Push(&operation.Mul[T]{
				Operand: moment,
				A:       momentum,
				B:       Momentum,
			})
			weight, err := p.Weights.GetOperand(i, j)
			if err != nil {
				return err
			}
			new_weight, err := p.NWeights.GetOperand(i, j)
			if err != nil {
				return err
			}
			ctx.Push(&operation.Sum[T]{
				Operand: new_weight,
				Args:    []*operation.Operand[T]{weight, moment, delta},
			})
		}
	}

	if p.PreLayer != nil {
		return p.PreLayer.BuildBackpropagation(ctx, Alpha, Momentum)
	}

	return nil
}

func (p *Dense[T]) Fit() error {
	if p.PreLayer != nil {
		p.PreLayer.Fit()
	}

	if !p.Trainable {
		return nil
	}
	err := p.Weights.LoadFromTensor(p.NWeights)
	if err != nil {
		return err
	}
	return p.Bias.LoadFromTensor(p.NBias)
}

func (p *Dense[T]) SetTrainable(t bool) {
	p.Trainable = t
}
