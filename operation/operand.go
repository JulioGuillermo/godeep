package operation

import (
	"github.com/julioguillermo/neuralnetwork/v2/types"
)

type Operand[T types.Number] struct {
	Value T
	// mutex sync.Mutex
}

//func (p *Operand[_]) Lock() {
//	p.mutex.Lock()
//}
//
//func (p *Operand[_]) Unlock() {
//	p.mutex.Unlock()
//}

//func (p *Operand[T]) Set(s T) {
//	p.Value = s
//	// p.mutex.Unlock()
//}
//
//func (p *Operand[T]) Get() T {
//	// p.mutex.Lock()
//	// defer p.mutex.Unlock()
//	return p.Value
//}
