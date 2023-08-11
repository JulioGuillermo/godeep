package layer

import (
	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

type Dense[T types.Number] struct {
	Base[T]
}

func NewDense[T types.Number](outs uint, act activation.Activation[T]) Layer[T] {
	outputs := tensor.NewZeros[T](outs)

	dense := &Dense[T]{}
	dense.Output = outputs
	dense.Activation = act
	dense.Trainable = true

	return dense
}

func NewInDense[T types.Number](ins, outs uint, act activation.Activation[T]) Layer[T] {
	inputs := tensor.NewZeros[T](ins)
	outputs := tensor.NewZeros[T](outs)

	dense := &Dense[T]{}
	dense.Input = inputs
	dense.Output = outputs
	dense.Activation = act
	dense.Trainable = true

	return dense
}

func (p *Dense[T]) Build() error {
	if p.CheckB() {
		return nil
	}

	return p.PreBuild()
}

func (p *Dense[T]) BuildFeedforward(ctx *context.Context) error {
	if p.CheckFF() {
		return nil
	}
	err := p.PreBuildFeedforward(ctx)
	if err != nil {
		return err
	}

	if p.Input == nil {
		return errors.FmtNeuralError("Invalid layer input => nil")
	}
	inputs := p.Input.GetSize()
	outputs := p.Output.GetSize()

	p.Neta = tensor.NewZeros[T](outputs)
	p.Dif = tensor.NewZeros[T](outputs)

	p.Bias = tensor.NewNormRand[T](outputs)
	p.Weights = tensor.NewNormRand[T](outputs, inputs)

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
	if p.CheckBP() {
		return nil
	}

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

	//mulDifAlpha := tensor.MulScalar(p.Dif, Alpha)
	//mulBiasMomM := tensor.MulScalar(p.MBias, Momentum)
	//deltaBias := tensor.Add(mulDifAlpha, mulBiasMomM)
	//p.NBias = tensor.Add(p.Bias, deltaBias)
	//err := p.NBias.BuildGraph(ctx)
	//if err != nil {
	//	return err
	//}
	//err = tensor.Transfer(ctx, mulDifAlpha, p.MBias)
	//if err != nil {
	//	return err
	//}

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

	return p.PostBuildBackpropagation(ctx, Alpha, Momentum)
}
