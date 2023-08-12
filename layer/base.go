package layer

import (
	"fmt"
	"strings"

	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
	"github.com/julioguillermo/godeep/types"
)

type Base[T types.Number] struct {
	Type  string
	Index uint

	Weights tensor.Tensor[T]
	Bias    tensor.Tensor[T]

	NWeights tensor.Tensor[T]
	MWeights tensor.Tensor[T]
	NBias    tensor.Tensor[T]
	MBias    tensor.Tensor[T]
	Alpha    *number.Scalar[T]
	Momentum *number.Scalar[T]

	PreLayer Layer[T]

	Input tensor.Tensor[T]

	Neta   tensor.Tensor[T]
	Output tensor.Tensor[T]
	Dif    tensor.Tensor[T]
	Ref    *number.Scalar[T]

	Activation activation.Activation[T]

	Trainable bool

	build     bool
	ffBuilded bool
	bpBuilded bool
	printed   bool
}

//func (p *Base[T]) Build() error {
//	return errors.FmtNeuralError("Base layer do not implement Build() method")
//}
//
//func (p *Base[T]) BuildFeedforward(ctx *context.Context) error {
//	return errors.FmtNeuralError("Base layer do not implement BuildFeedforward() method")
//}
//
//func (p *Base[T]) BuildBackpropagation(
//	ctx *context.Context,
//	alpha, momentum *operation.Operand[T],
//) error {
//	return errors.FmtNeuralError("Base layer do not implement BuildBackpropagation() method")
//}

func (p *Base[T]) Error(msg string) error {
	return errors.FmtNeuralError("Layer[%d] type %s: %s", p.Index, p.Type, msg)
}

func (p *Base[T]) CheckB() bool {
	if p.build {
		return true
	}
	p.build = true
	return false
}

func (p *Base[T]) PreBuild() error {
	if p.Ref == nil {
		p.Ref = &number.Scalar[T]{}
	}
	if p.PreLayer != nil {
		p.PreLayer.GetRef().Value++
		count, err := p.PreLayer.Build()
		if err != nil {
			return err
		}
		p.Index = count + 1
	}
	if p.Ref.Value == 0 {
		return p.Error("Unconnected layer")
	}
	return nil
}

func (p *Base[T]) CheckFF() bool {
	if p.ffBuilded || p.Ref.Value == 0 {
		return true
	}
	p.ffBuilded = true
	return false
}

func (p *Base[T]) PreBuildFeedforward(ctx *context.Context) error {
	if p.PreLayer != nil {
		err := p.PreLayer.BuildFeedforward(ctx)
		if err != nil {
			return err
		}
		p.Input = p.PreLayer.GetOutputs()
	}
	return nil
}

func (p *Base[T]) CheckBP() bool {
	if p.bpBuilded || p.Ref.Value == 0 {
		return true
	}
	p.bpBuilded = true
	return false
}

func (p *Base[T]) PostBuildBackpropagation(
	ctx *context.Context,
	alpha, momentum *number.Scalar[T],
) error {
	if p.PreLayer != nil {
		err := p.PreLayer.BuildBackpropagation(ctx, alpha, momentum)
		if err != nil {
			return err
		}
	}
	return nil
}

func (p *Base[T]) GetIndex() uint {
	return p.Index
}

func (p *Base[T]) GetInputs() tensor.Tensor[T] {
	return p.Input
}

func (p *Base[T]) GetOutputs() tensor.Tensor[T] {
	return p.Output
}

func (p *Base[T]) GetNetas() tensor.Tensor[T] {
	return p.Neta
}

func (p *Base[T]) GetDif() tensor.Tensor[T] {
	return p.Dif
}

func (p *Base[T]) GetRef() *number.Scalar[T] {
	if p.Ref == nil {
		p.Ref = &number.Scalar[T]{}
	}
	return p.Ref
}

func (p *Base[T]) GetActivation() activation.Activation[T] {
	return p.Activation
}

func (p *Base[T]) GetPrelayer() Layer[T] {
	return p.PreLayer
}

func (p *Base[T]) Connect(l Layer[T]) {
	p.PreLayer = l
}

func (p *Base[T]) Fit() error {
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

func (p *Base[T]) SetTrainable(t bool) {
	p.Trainable = t
}

func (p *Base[T]) ResetFit(ctx *context.Context) error {
	if p.PreLayer != nil {
		err := p.PreLayer.ResetFit(ctx)
		if err != nil {
			return err
		}
	}
	return tensor.Fill(ctx, p.Dif, &number.Scalar[T]{})
}

func (p *Base[T]) Reset(ctx *context.Context) error {
	if p.PreLayer != nil {
		err := p.PreLayer.Reset(ctx)
		if err != nil {
			return err
		}
	}
	return nil
}

func (p *Base[T]) ResetPrinted() {
	p.printed = false

	if p.PreLayer != nil {
		p.PreLayer.ResetPrinted()
	}
}

func (p *Base[T]) PushToString(sb *strings.Builder) {
	if p.printed {
		return
	}
	if p.PreLayer != nil {
		p.PreLayer.PushToString(sb)
		sb.WriteString(
			fmt.Sprintf(
				"<Layer[%d] (%d) => %s: O%s>\n",
				p.Index,
				p.PreLayer.GetIndex(),
				p.Type,
				tools.ShapeStr(p.Output.GetShape()),
			),
		)
		return
	}
	sb.WriteString(
		fmt.Sprintf(
			"<Layer[%d] => %s: O%s>\n",
			p.Index,
			p.Type,
			tools.ShapeStr(p.Output.GetShape()),
		),
	)
}
