package layer

import (
	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
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
	dense.Type = "Dense"
	dense.Output = outputs
	dense.Activation = act
	dense.Trainable = true

	return dense
}

func NewInDense[T types.Number](ins, outs uint, act activation.Activation[T]) Layer[T] {
	inputs := tensor.NewZeros[T](ins)
	outputs := tensor.NewZeros[T](outs)

	dense := &Dense[T]{}
	dense.Type = "Dense"
	dense.Input = inputs
	dense.Output = outputs
	dense.Activation = act
	dense.Trainable = true

	return dense
}

func (p *Dense[T]) Build() (uint, error) {
	if p.CheckB() {
		return p.Index, nil
	}

	return p.Index, p.PreBuild()
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
		return p.Error("Invalid input => nil")
	}

	inputs := p.Input.GetSize()
	outputs := p.Output.GetSize()

	p.Neta = tensor.NewZeros[T](outputs)
	p.Dif = tensor.NewZeros[T](outputs)

	p.Bias = tensor.NewNormRand[T](outputs)
	p.Weights = tensor.NewNormRand[T](outputs, inputs)

	for i := uint(0); i < outputs; i++ {
		ops := make([]*number.Scalar[T], inputs+1)
		bias, err := p.Bias.GetOperand(i)
		if err != nil {
			return err
		}
		ops[inputs] = bias

		for j := uint(0); j < inputs; j++ {
			mul_iw := &number.Scalar[T]{}
			ops[j] = mul_iw
			weight, err := p.Weights.GetOperand(i, j)
			if err != nil {
				return err
			}
			input := p.Input.GetOperands()[j]
			ctx.Push(&operation.Mul[T]{
				Scalar: mul_iw,
				A:      input,
				B:      weight,
			})
		}

		neta, err := p.Neta.GetOperand(i)
		if err != nil {
			return err
		}
		ctx.Push(&operation.Sum[T]{
			Scalar: neta,
			Args:   ops,
		})

		output, err := p.Output.GetOperand(i)
		if err != nil {
			return err
		}
		ctx.Push(&operation.Func[T]{
			Scalar: output,
			O:      neta,
			F:      p.Activation.Activate,
		})
	}

	return nil
}

func (p *Dense[T]) BuildBackpropagation(
	ctx *context.Context,
	Alpha, Momentum *number.Scalar[T],
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

	Dif := p.Dif
	if p.Ref.Value > 1 {
		Dif = tensor.DivScalar(Dif, p.Ref)
	}
	err := Dif.BuildGraph(ctx)
	if err != nil {
		return err
	}

	if p.PreLayer != nil {
		der := tensor.Activate(p.PreLayer.GetNetas(), p.PreLayer.GetActivation().Derive)
		err := der.BuildGraph(ctx)
		if err != nil {
			return err
		}

		for i := uint(0); i < inputs; i++ {
			ops := make([]*number.Scalar[T], outputs)
			for j := uint(0); j < outputs; j++ {
				dif, err := Dif.GetOperand(j)
				if err != nil {
					return err
				}
				w, err := p.Weights.GetOperand(j, i)
				if err != nil {
					return err
				}
				o := &number.Scalar[T]{}
				ctx.Push(&operation.Mul[T]{
					Scalar: o,
					A:      dif,
					B:      w,
				})
				ops[j] = o
			}
			sum := &number.Scalar[T]{}
			ctx.Push(&operation.Sum[T]{
				Scalar: sum,
				Args:   ops,
			})
			d := der.GetOperands()[i]
			nd := &number.Scalar[T]{}
			pd := p.PreLayer.GetDif().GetOperands()[i]
			ctx.Push(&operation.Mul[T]{
				Scalar: nd,
				A:      sum,
				B:      d,
			})
			npd := &number.Scalar[T]{}
			ctx.Push(&operation.Add[T]{
				Scalar: npd,
				A:      pd,
				B:      nd,
			})
			ctx.Push(&operation.Set[T]{
				Scalar: pd,
				O:      npd,
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
		dif, err := Dif.GetOperand(i)
		if err != nil {
			return err
		}
		mul_dif_alpha := &number.Scalar[T]{}
		ctx.Push(&operation.Mul[T]{
			Scalar: mul_dif_alpha,
			A:      dif,
			B:      Alpha,
		})
		momentum, err := p.MBias.GetOperand(i)
		if err != nil {
			return err
		}
		moment := &number.Scalar[T]{}
		ctx.Push(&operation.Mul[T]{
			Scalar: moment,
			A:      momentum,
			B:      Momentum,
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
			Scalar: new_bias,
			Args:   []*number.Scalar[T]{bias, moment, mul_dif_alpha},
		})

		for j := uint(0); j < inputs; j++ {
			input := p.Input.GetOperands()[j]
			momentum, err := p.MWeights.GetOperand(i, j)
			if err != nil {
				return err
			}
			delta := &number.Scalar[T]{}
			ctx.Push(&operation.Mul[T]{
				Scalar: delta,
				A:      mul_dif_alpha,
				B:      input,
			})
			moment := &number.Scalar[T]{}
			ctx.Push(&operation.Mul[T]{
				Scalar: moment,
				A:      momentum,
				B:      Momentum,
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
				Scalar: new_weight,
				Args:   []*number.Scalar[T]{weight, moment, delta},
			})
		}
	}

	return p.PostBuildBackpropagation(ctx, Alpha, Momentum)
}
