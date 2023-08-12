package img

import (
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"os"

	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

func ToImg[T types.Number](img tensor.Tensor[T]) (image.Image, error) {
	shape := img.GetShape()
	if len(shape) != 3 {
		return nil, errors.FmtNeuralError("The given tensor is not an image")
	}

	newImg := image.NewRGBA(image.Rect(0, 0, int(shape[1]), int(shape[2])))

	var i uint
	var j uint

	var r T
	var g T
	var b T
	var a T
	var c color.RGBA

	var err error
	for i = 0; i < shape[1]; i++ {
		for j = 0; j < shape[2]; j++ {
			r, err = img.Get(0, i, j)
			if err != nil {
				return nil, err
			}
			g, err = img.Get(1, i, j)
			if err != nil {
				return nil, err
			}
			b, err = img.Get(2, i, j)
			if err != nil {
				return nil, err
			}
			a, err = img.Get(3, i, j)
			if err != nil {
				return nil, err
			}

			c = color.RGBA{
				R: uint8(r * 255),
				G: uint8(g * 255),
				B: uint8(b * 255),
				A: uint8(a * 255),
			}

			newImg.SetRGBA(int(i), int(j), c)
		}
	}

	return newImg, nil
}

func PngSave[T types.Number](img tensor.Tensor[T], p string) error {
	newImg, err := ToImg[T](img)
	if err != nil {
		return err
	}

	file, err := os.Create(p)
	if err != nil {
		return err
	}
	defer file.Close()

	return png.Encode(file, newImg)
}

func JpgSave[T types.Number](img tensor.Tensor[T], p string, quality int) error {
	newImg, err := ToImg[T](img)
	if err != nil {
		return err
	}

	file, err := os.Create(p)
	if err != nil {
		return err
	}
	defer file.Close()

	opt := &jpeg.Options{Quality: quality}
	return jpeg.Encode(file, newImg, opt)
}
