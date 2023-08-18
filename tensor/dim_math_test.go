package tensor_test

import (
	"testing"

	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/tensor"
)

func TestDimSum(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.DSum[float32](m, 1)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			f, err := r.Get(i, j)
			if err != nil {
				t.Fatal(err)
			}
			sum := float32(0)
			for k := uint(0); k < shape[2]; k++ {
				n, err := m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				sum += n
			}

			if f != sum {
				t.Fatal(f, "!=", sum)
			}
		}
	}
}

func TestDimAvg(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.DAvg[float32](m, 1)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			f, err := r.Get(i, j)
			if err != nil {
				t.Fatal(err)
			}
			sum := float32(0)
			for k := uint(0); k < shape[2]; k++ {
				n, err := m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				sum += n
			}
			sum /= float32(shape[2])

			if f != sum {
				t.Fatal(f, "!=", sum)
			}
		}
	}
}

func TestDimMax(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.DMax[float32](m, 1)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			f, err := r.Get(i, j)
			if err != nil {
				t.Fatal(err)
			}
			n, err := m.Get(i, j, 0)
			if err != nil {
				t.Fatal(err)
			}
			sum := n
			for k := uint(1); k < shape[2]; k++ {
				n, err = m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				if sum < n {
					sum = n
				}
			}

			if f != sum {
				t.Fatal(f, "!=", sum)
			}
		}
	}
}

func TestDimMin(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r := tensor.DMin[float32](m, 1)

	g, err := graph.NewGraph(r)
	if err != nil {
		t.Fatal(err)
	}

	g.Exec()

	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			f, err := r.Get(i, j)
			if err != nil {
				t.Fatal(err)
			}
			n, err := m.Get(i, j, 0)
			if err != nil {
				t.Fatal(err)
			}
			sum := n
			for k := uint(1); k < shape[2]; k++ {
				n, err = m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				if sum > n {
					sum = n
				}
			}

			if f != sum {
				t.Fatal(f, "!=", sum)
			}
		}
	}
}

func TestHotDimSum(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r, err := m.DSum(1)
	if err != nil {
		t.Fatal(err)
	}

	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			f, err := r.Get(i, j)
			if err != nil {
				t.Fatal(err)
			}
			sum := float32(0)
			for k := uint(0); k < shape[2]; k++ {
				n, err := m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				sum += n
			}

			if f != sum {
				t.Fatal(f, "!=", sum)
			}
		}
	}
}

func TestHotDimSum2(t *testing.T) {
	shape := []uint{50, 50, 20, 30}

	m := tensor.NewNormRand[float32](shape...)
	r, err := m.DSum(1)
	if err != nil {
		t.Fatal(err)
	}

	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			f, err := r.Get(i, j)
			if err != nil {
				t.Fatal(err)
			}
			sum := float32(0)
			for k := uint(0); k < shape[2]; k++ {
				for l := uint(0); l < shape[3]; l++ {
					n, err := m.Get(i, j, k, l)
					if err != nil {
						t.Fatal(err)
					}
					sum += n
				}
			}

			if f != sum {
				t.Fatal(f, "!=", sum)
			}
		}
	}
}

func TestHotDimAvg(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r, err := m.DAvg(1)
	if err != nil {
		t.Fatal(err)
	}

	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			f, err := r.Get(i, j)
			if err != nil {
				t.Fatal(err)
			}
			sum := float32(0)
			for k := uint(0); k < shape[2]; k++ {
				n, err := m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				sum += n
			}
			sum /= float32(shape[2])

			if f != sum {
				t.Fatal(f, "!=", sum)
			}
		}
	}
}

func TestHotDimMax(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r, err := m.DMax(1)
	if err != nil {
		t.Fatal(err)
	}

	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			f, err := r.Get(i, j)
			if err != nil {
				t.Fatal(err)
			}
			n, err := m.Get(i, j, 0)
			if err != nil {
				t.Fatal(err)
			}
			sum := n
			for k := uint(1); k < shape[2]; k++ {
				n, err = m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				if sum < n {
					sum = n
				}
			}

			if f != sum {
				t.Fatal(f, "!=", sum)
			}
		}
	}
}

func TestHotDimMin(t *testing.T) {
	shape := []uint{50, 50, 20}

	m := tensor.NewNormRand[float32](shape...)
	r, err := m.DMin(1)
	if err != nil {
		t.Fatal(err)
	}

	for i := uint(0); i < shape[0]; i++ {
		for j := uint(0); j < shape[1]; j++ {
			f, err := r.Get(i, j)
			if err != nil {
				t.Fatal(err)
			}
			n, err := m.Get(i, j, 0)
			if err != nil {
				t.Fatal(err)
			}
			sum := n
			for k := uint(1); k < shape[2]; k++ {
				n, err = m.Get(i, j, k)
				if err != nil {
					t.Fatal(err)
				}
				if sum > n {
					sum = n
				}
			}

			if f != sum {
				t.Fatal(f, "!=", sum)
			}
		}
	}
}
