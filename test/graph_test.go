package test

import (
	"testing"

	"github.com/julioguillermo/neuralnetwork/v2/graph"
	"github.com/julioguillermo/neuralnetwork/v2/tensor"
)

func TestGraph(t *testing.T) {
	m1 := tensor.NewRand[float32](-1, 1, 50, 50, 20)
	m2 := tensor.NewRand[float32](-1, 1, 50, 50, 20)

	sum := tensor.Add[float32](m1, m2)
	mul := tensor.Mul[float32](m1, m2)
	sub := tensor.Sub(mul, sum)

	g, err := graph.NewGraph(sub)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()
	t.Log(sub)
}

//func TestGraphFullAsync(t *testing.T) {
//	m1 := tensor.NewRand[float32](-1, 1, 50, 50, 20)
//	m2 := tensor.NewRand[float32](-1, 1, 50, 50, 20)
//
//	sum := tensor.Add[float32](m1, m2)
//	mul := tensor.Mul[float32](m1, m2)
//	sub := tensor.Sub(mul, sum)
//
//	g, err := graph.NewGraph(sub)
//	if err != nil {
//		t.Fatal(err)
//	}
//
//	g.FullAsyncExec()
//	g.Exec()
//	t.Log(sub)
//}
//
//func TestGraphAsync(t *testing.T) {
//	m1 := tensor.NewRand[float32](-1, 1, 50, 50, 20)
//	m2 := tensor.NewRand[float32](-1, 1, 50, 50, 20)
//
//	sum := tensor.Add(m1, m2)
//	mul := tensor.Mul(m1, m2)
//	sub := tensor.Sub(mul, sum)
//
//	g, err := graph.NewGraph(sub)
//	if err != nil {
//		t.Fatal(err)
//	}
//
//	g.AsyncExec(100)
//	g.Exec()
//	t.Log(sub)
//}
