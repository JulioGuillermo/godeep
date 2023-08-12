package operation

import (
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/types"
)

type Max[T types.Number] struct {
	*number.Scalar[T]
	Args []*number.Scalar[T]
}

func (p *Max[_]) Cal() {
	num := p.Args[0].Value
	// num := p.Args[0].Get()
	size := len(p.Args)
	// var n T
	for i := 1; i < size; i++ {
		// n = p.Args[i].Get()
		if num < p.Args[i].Value {
			num = p.Args[i].Value
		}
	}
	p.Value = num
	// p.Set(num)
}

type Min[T types.Number] struct {
	*number.Scalar[T]
	Args []*number.Scalar[T]
}

func (p *Min[_]) Cal() {
	// num := p.Args[0].Get()
	num := p.Args[0].Value
	size := len(p.Args)
	// var n T
	for i := 1; i < size; i++ {
		// n = p.Args[i].Get()
		if num > p.Args[i].Value {
			num = p.Args[i].Value
		}
	}
	p.Value = num
	// p.Set(num)
}
