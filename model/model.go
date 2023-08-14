package model

import (
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"

	"golang.org/x/crypto/ssh/terminal"

	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/layer"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
	"github.com/julioguillermo/godeep/types"
)

type Model[T types.Number] struct {
	FirstLayer layer.Layer[T]
	LastLayer  layer.Layer[T]

	input    tensor.Tensor[T]
	output   tensor.Tensor[T]
	target   tensor.Tensor[T]
	alpha    *number.Scalar[T]
	momentum *number.Scalar[T]
	loss     *number.Scalar[T]

	feedForward     *graph.Graph
	backPropagation *graph.Graph
	resetFit        *graph.Graph
	reset           *graph.Graph
}

func NewModel[T types.Number]() *Model[T] {
	return &Model[T]{}
}

func FromInOut[T types.Number](in, out layer.Layer[T]) (*Model[T], error) {
	return &Model[T]{
		FirstLayer: in,
		LastLayer:  out,
	}, nil
}

func (p *Model[T]) Push(l layer.Layer[T]) *Model[T] {
	if p.FirstLayer == nil {
		p.FirstLayer = l
		p.LastLayer = l
		return p
	}
	l.Connect(p.LastLayer)
	p.LastLayer = l
	return p
}

func (p *Model[T]) Compile() error {
	p.LastLayer.GetRef().Value++
	_, err := p.Build()
	if err != nil {
		return err
	}

	ctx := &context.Context{}
	err = p.BuildFeedforward(ctx)
	if err != nil {
		return err
	}

	ff, err := graph.NewGraphFrom[T](ctx)
	if err != nil {
		return err
	}

	ctx = &context.Context{}
	err = p.LastLayer.Reset(ctx)
	if err != nil {
		return err
	}
	reset, err := graph.NewGraphFrom[T](ctx)
	if err != nil {
		return err
	}

	p.feedForward = ff
	p.reset = reset
	p.input = p.FirstLayer.GetInputs()
	p.output = p.LastLayer.GetOutputs()
	p.target = p.output.Copy()
	return nil
}

func (p *Model[T]) buildBackPropagation() error {
	ctx := &context.Context{}

	p.alpha = &number.Scalar[T]{}
	p.momentum = &number.Scalar[T]{}

	dif := p.LastLayer.GetDif().SetBuild(false)
	p.target = tensor.NewZeros[T](p.output.GetShape()...)
	out_ops := p.output.GetOperands()
	tar_ops := p.target.GetOperands()
	dif_ops := dif.GetOperands()
	for i := range tar_ops {
		ctx.Push(&operation.Sub[T]{
			Scalar: dif_ops[i],
			A:      tar_ops[i],
			B:      out_ops[i],
		})
	}

	loss := tensor.Abs(dif)
	loss = tensor.Sum(loss)
	loss.BuildGraph(ctx)
	p.loss = loss.GetOperands()[0]

	err := p.LastLayer.BuildBackpropagation(ctx, p.alpha, p.momentum)
	if err != nil {
		return err
	}

	bp, err := graph.NewGraphFrom[T](ctx)
	if err != nil {
		return err
	}

	ctx = &context.Context{}
	err = p.LastLayer.ResetFit(ctx)
	if err != nil {
		return err
	}
	rf, err := graph.NewGraphFrom[T](ctx)
	if err != nil {
		return err
	}

	p.backPropagation = bp
	p.resetFit = rf
	return nil
}

func (p *Model[T]) ResetIS() error {
	if p.feedForward == nil {
		err := p.Compile()
		if err != nil {
			return err
		}
	}
	p.reset.Exec()
	return nil
}

func (p *Model[T]) Predict(t tensor.Tensor[T]) (tensor.Tensor[T], error) {
	if p.feedForward == nil {
		err := p.Compile()
		if err != nil {
			return nil, err
		}
	}
	err := p.input.LoadFromTensor(t)
	if err != nil {
		return nil, err
	}
	p.feedForward.Exec()
	return p.output.Copy(), nil
}

func (p *Model[T]) fit(x, y tensor.Tensor[T]) error {
	p.resetFit.Exec()

	err := p.input.LoadFromTensor(x)
	if err != nil {
		return err
	}
	err = p.target.LoadFromTensor(y)
	if err != nil {
		return err
	}

	p.feedForward.Exec()
	p.backPropagation.Exec()

	return p.LastLayer.Fit()
}

func (p *Model[T]) TrainOne(
	input, target tensor.Tensor[T],
	alpha, momentum T,
) (T, error) {
	if p.backPropagation == nil {
		err := p.buildBackPropagation()
		if err != nil {
			return -1, err
		}
	}
	p.alpha.Value = alpha
	p.momentum.Value = momentum
	err := p.fit(input, target)
	if err != nil {
		return -1, err
	}
	return p.loss.Value, err
}

func (p *Model[T]) Train(
	inputs, targets []tensor.Tensor[T],
	epochs, batch uint,
	alpha, momentum T,
) error {
	if len(inputs) != len(targets) {
		return errors.FmtNeuralError(
			"Fail to fit the model, the number of inputs %d does not match with the number of targets %d",
			len(inputs),
			len(targets),
		)
	}
	if p.backPropagation == nil {
		err := p.buildBackPropagation()
		if err != nil {
			return err
		}
	}
	p.alpha.Value = alpha
	p.momentum.Value = momentum
	if batch == 0 {
		batch = uint(len(inputs))
	}
	start := time.Now()
	delta := time.Millisecond * 300

	width, _, err := terminal.GetSize(int(os.Stdout.Fd()))
	vervose := err == nil

	sE := fmt.Sprint(epochs)
	sB := fmt.Sprint(batch)
	lE := len(sE)
	lB := len(sB)

	for i := uint(0); i < epochs; i++ {
		for j := uint(1); j <= batch; j++ {
			index := rand.Intn(len(inputs))
			err := p.fit(inputs[index], targets[index])
			if err != nil {
				return err
			}
			if vervose && time.Since(start) > delta {
				start = time.Now()
				prog := float64(i*batch+j) / float64(epochs*batch)

				sProg := fmt.Sprintf("%.2f%%", prog*100)
				sI := fmt.Sprint(i)
				sJ := fmt.Sprint(j)
				sLoss := fmt.Sprintf("%.8f", float64(p.loss.Value))
				lLoss := 10 - len(sLoss)
				if lLoss < 0 {
					lLoss = 0
				}

				info := fmt.Sprintf(
					"\r[%s%s] %s%s / %s => <%s%s / %s> %s%s",
					strings.Repeat(" ", 7-len(sProg)),
					sProg,

					strings.Repeat(" ", lE-len(sI)),
					sI,

					sE,

					strings.Repeat(" ", lB-len(sJ)),
					sJ,

					sB,

					strings.Repeat(" ", lLoss),
					sLoss,
				)

				fmt.Print(info, " ", tools.Bar(prog, width-len(info)-1))
			}
		}
		//if i%100 == 0 {
		//}
	}

	info := fmt.Sprintf(
		"\r[100.00%%] %d / %d => %f",
		epochs,
		epochs,
		float64(p.loss.Value),
	)
	fmt.Println(info, tools.Bar(1, width-len(info)-1))

	return nil
}

func (p *Model[T]) String() string {
	p.ResetPrinted()

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("<Model => [\n"))
	p.PushToString(&sb)
	sb.WriteString("]>\n")
	return sb.String()
}
