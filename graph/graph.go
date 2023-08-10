package graph

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

type Graph struct {
	Operations []operation.Operation
}

func NewGraphFrom[T types.Number](ctx *context.Context) (*Graph, error) {
	return &Graph{ctx.GetOps()}, nil
}

func NewGraph[T types.Number](t tensor.Tensor[T]) (*Graph, error) {
	ctx := context.NewCTX()
	err := t.BuildGraph(ctx)
	if err != nil {
		return nil, err
	}
	return &Graph{ctx.GetOps()}, nil
}

func (p *Graph) Exec() {
	//for _, o := range p.Operations {
	//	o.Lock()
	//}
	for _, o := range p.Operations {
		o.Cal()
	}
}

//func lock(o operation.Operation, wg *sync.WaitGroup) {
//	o.Lock()
//	wg.Done()
//}
//
//func cal(o operation.Operation, wg *sync.WaitGroup) {
//	o.Cal()
//	wg.Done()
//}
//
//func (p *Graph) FullAsyncExec() {
//	var wg sync.WaitGroup
//	for _, o := range p.Operations {
//		wg.Add(1)
//		go lock(o, &wg)
//	}
//	wg.Wait()
//	for _, o := range p.Operations {
//		wg.Add(1)
//		go cal(o, &wg)
//	}
//	wg.Wait()
//}
//
//func (p *Graph) asyncLock(ch chan int, wg *sync.WaitGroup) {
//	for i := range ch {
//		p.Operations[i].Lock()
//		wg.Done()
//	}
//}
//
//func (p *Graph) asyncExec(ch chan int, wg *sync.WaitGroup) {
//	for i := range ch {
//		p.Operations[i].Cal()
//		wg.Done()
//	}
//}
//
//func (p *Graph) AsyncExec(async int) {
//	var wg sync.WaitGroup
//	ch := make(chan int, async)
//
//	for i := 0; i < async; i++ {
//		go p.asyncLock(ch, &wg)
//	}
//	for i := range p.Operations {
//		wg.Add(1)
//		ch <- i
//	}
//	wg.Wait()
//	close(ch)
//
//	ch = make(chan int, async)
//	for i := 0; i < async; i++ {
//		go p.asyncExec(ch, &wg)
//	}
//	for i := range p.Operations {
//		wg.Add(1)
//		ch <- i
//	}
//	wg.Wait()
//	close(ch)
//}
