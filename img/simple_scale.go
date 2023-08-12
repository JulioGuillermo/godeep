package img

import (
	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

func SimpleScale[T types.Number](
	img tensor.Tensor[T],
	width, height uint,
) (tensor.Tensor[T], error) {
	shape := img.GetShape()
	if len(shape) != 3 {
		return nil, errors.FmtNeuralError("The given tensor is not an image")
	}

	newImg := tensor.NewZeros[T](shape[0], width, height)
	scalX := float64(shape[1]) / float64(width)
	scalY := float64(shape[2]) / float64(height)

	var X uint
	var Y uint
	var x uint
	var y uint
	var d uint
	var c T
	var err error
	for x = 0; x < width; x++ {
		for y = 0; y < height; y++ {
			X = uint(float64(x) * scalX)
			Y = uint(float64(y) * scalY)
			for d = 0; d < shape[0]; d++ {
				c, err = img.Get(d, X, Y)
				if err != nil {
					return nil, err
				}
				newImg.Set(c, d, x, y)
			}
		}
	}

	return newImg, nil
}
