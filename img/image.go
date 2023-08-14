package img

import (
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

const MaxColor = 65535

func LoadImage[T types.Number](path string) (tensor.Tensor[T], error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	size := img.Bounds().Size()
	width := size.X
	height := size.Y

	t := tensor.NewZeros[T](4, uint(width), uint(height))
	for i := 0; i < width; i++ {
		for j := 0; j < height; j++ {
			c := img.At(i, j)
			r, g, b, a := c.RGBA()
			t.Set(T(float64(r)/MaxColor), 0, uint(i), uint(j))
			t.Set(T(float64(g)/MaxColor), 1, uint(i), uint(j))
			t.Set(T(float64(b)/MaxColor), 2, uint(i), uint(j))
			t.Set(T(float64(a)/MaxColor), 3, uint(i), uint(j))
		}
	}

	return t, nil
}

func LoadSimpleScaled[T types.Number](path string, width, height uint) (tensor.Tensor[T], error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	size := img.Bounds().Size()

	scalX := float64(size.X) / float64(width)
	scalY := float64(size.Y) / float64(height)
	t := tensor.NewZeros[T](4, width, height)

	var i uint
	var j uint
	var X int
	var Y int
	for i = 0; i < width; i++ {
		for j = 0; j < height; j++ {
			X = int(float64(i) * scalX)
			Y = int(float64(j) * scalY)
			c := img.At(X, Y)
			r, g, b, a := c.RGBA()
			t.Set(T(float64(r)/MaxColor), 0, uint(i), uint(j))
			t.Set(T(float64(g)/MaxColor), 1, uint(i), uint(j))
			t.Set(T(float64(b)/MaxColor), 2, uint(i), uint(j))
			t.Set(T(float64(a)/MaxColor), 3, uint(i), uint(j))
		}
	}

	return t, nil
}
