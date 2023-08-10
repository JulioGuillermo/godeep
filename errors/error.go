package errors

import "fmt"

type NeuralError string

func FmtNeuralError(f string, args ...any) error {
	return NeuralError(fmt.Sprintf(f, args...))
}

func (p NeuralError) Error() string {
	return string(p)
}
