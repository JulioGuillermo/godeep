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

	Input    tensor.Tensor[T]
	Output   tensor.Tensor[T]
	Target   tensor.Tensor[T]
	Alpha    *number.Scalar[T]
	Momentum *number.Scalar[T]
	Loss     *number.Scalar[T]

	GraphFeedForward     *graph.Graph
	GraphBackPropagation *graph.Graph
	GraphCalDif          *graph.Graph
	GraphResetFit        *graph.Graph
	GraphReset           *graph.Graph
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

func FromModels[T types.Number](models ...*Model[T]) (*Model[T], error) {
	if len(models) == 0 {
		return nil, errors.FmtNeuralError("Can not create a model from 0 models")
	}

	for i := 1; i < len(models); i++ {
		if models[i].FirstLayer.GetPrelayer() != models[i-1].LastLayer {
			models[i].FirstLayer.Connect(models[i-1].LastLayer)
		}
	}

	for i := 0; i < len(models); i++ {
		if models[i].GraphFeedForward == nil {
			models[i].Compile()
		}
	}

	for i := len(models) - 1; i >= 0; i-- {
		if models[i].GraphBackPropagation == nil {
			models[i].CompileBackPropagation()
		}
	}

	m := NewModel[T]()

	size := len(models)

	m.FirstLayer = models[0].FirstLayer
	m.Input = models[0].Input

	m.LastLayer = models[size-1].LastLayer
	m.Output = models[size-1].Output
	m.Target = models[size-1].Target
	m.Alpha = models[size-1].Alpha
	m.Momentum = models[size-1].Momentum
	m.Loss = models[size-1].Loss

	m.GraphFeedForward = graph.NewEmptyGraph[T]()
	for i, mod := range models {
		if mod.GraphFeedForward == nil {
			return nil, errors.FmtNeuralError(
				"Model %d: FeedForward is not compiled (FeedForward)",
				i,
			)
		}
		m.GraphFeedForward.AddFromGraph(mod.GraphFeedForward)
	}

	m.GraphReset = graph.NewEmptyGraph[T]()
	for i, mod := range models {
		if mod.GraphReset == nil {
			return nil, errors.FmtNeuralError("Model %d: FeedForward is not compiled (Reset)", i)
		}
		m.GraphReset.AddFromGraph(mod.GraphReset)
	}

	m.GraphResetFit = graph.NewEmptyGraph[T]()
	for i, mod := range models {
		if mod.GraphResetFit == nil {
			return nil, errors.FmtNeuralError(
				"Model %d: BackPropagation is not compiled (ResetFit)",
				i,
			)
		}
		m.GraphResetFit.AddFromGraph(mod.GraphResetFit)
	}

	m.GraphCalDif = graph.NewFromGraph[float32](models[size-1].GraphCalDif)
	if models[size-1].GraphCalDif == nil {
		return nil, errors.FmtNeuralError(
			"Last model %d: BackPropagation is not compiled (CalDif)",
			size-1,
		)
	}

	m.GraphBackPropagation = graph.NewEmptyGraph[T]()
	for i := size - 1; i >= 0; i-- {
		if models[i].GraphBackPropagation == nil {
			return nil, errors.FmtNeuralError(
				"Model %d: BackPropagation is not compiled (BackPropagation)",
				i,
			)
		}
		m.GraphBackPropagation.AddFromGraph(models[i].GraphBackPropagation)
	}

	return m, nil
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

	p.GraphFeedForward = ff
	p.GraphReset = reset
	p.Input = p.FirstLayer.GetInputs()
	p.Output = p.LastLayer.GetOutputs()
	p.Target = p.Output.Copy()
	return nil
}

func (p *Model[T]) CompileBackPropagation() error {
	ctx := &context.Context{}

	dif := p.LastLayer.GetDif().SetBuild(false)
	err := dif.BuildGraph(ctx)
	if err != nil {
		return err
	}
	p.Target = tensor.NewZeros[T](p.Output.GetShape()...)
	out_ops := p.Output.GetOperands()
	tar_ops := p.Target.GetOperands()
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
	p.Loss = loss.GetOperands()[0]

	cd, err := graph.NewGraphFrom[T](ctx)
	p.GraphCalDif = cd

	ctx = &context.Context{}

	p.Alpha = &number.Scalar[T]{}
	p.Momentum = &number.Scalar[T]{}

	p.LastLayer.GetDif().SetBuild(false)
	err = p.LastLayer.BuildBackpropagation(ctx, p.Alpha, p.Momentum)
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

	p.GraphBackPropagation = bp
	p.GraphResetFit = rf
	return nil
}

func (p *Model[T]) ResetIS() error {
	if p.GraphFeedForward == nil {
		err := p.Compile()
		if err != nil {
			return err
		}
	}
	p.GraphReset.Exec()
	return nil
}

func (p *Model[T]) Predict(t tensor.Tensor[T]) (tensor.Tensor[T], error) {
	if p.GraphFeedForward == nil {
		err := p.Compile()
		if err != nil {
			return nil, err
		}
	}
	err := p.Input.LoadFromTensor(t)
	if err != nil {
		return nil, err
	}
	p.GraphFeedForward.Exec()
	return p.Output.Copy(), nil
}

func (p *Model[T]) ExecFit(x, y tensor.Tensor[T]) error {
	p.GraphResetFit.Exec()

	err := p.Input.LoadFromTensor(x)
	if err != nil {
		return err
	}
	err = p.Target.LoadFromTensor(y)
	if err != nil {
		return err
	}

	p.GraphFeedForward.Exec()
	p.GraphCalDif.Exec()
	p.GraphBackPropagation.Exec()

	return p.LastLayer.Fit()
}

func (p *Model[T]) TrainOne(
	input, target tensor.Tensor[T],
	alpha, momentum T,
) (T, error) {
	if p.GraphBackPropagation == nil {
		err := p.CompileBackPropagation()
		if err != nil {
			return -1, err
		}
	}
	p.Alpha.Value = alpha
	p.Momentum.Value = momentum
	err := p.ExecFit(input, target)
	if err != nil {
		return -1, err
	}
	return p.Loss.Value, err
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
	if p.GraphBackPropagation == nil {
		err := p.CompileBackPropagation()
		if err != nil {
			return err
		}
	}
	p.Alpha.Value = alpha
	p.Momentum.Value = momentum
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
			err := p.ExecFit(inputs[index], targets[index])
			if err != nil {
				return err
			}
			if vervose && time.Since(start) > delta {
				start = time.Now()
				prog := float64(i*batch+j) / float64(epochs*batch)

				sProg := fmt.Sprintf("%.2f%%", prog*100)
				sI := fmt.Sprint(i)
				sJ := fmt.Sprint(j)
				sLoss := fmt.Sprintf("%.8f", float64(p.Loss.Value))
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
		float64(p.Loss.Value),
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
