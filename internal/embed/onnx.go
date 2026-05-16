package embed

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

const EmbeddingDim = 384

type ONNXEmbedder struct {
	session *ort.AdvancedSession
	inputs  []ort.Value
	outputs []ort.Value
}

func NewONNXEmbedder(modelPath, onnxLibPath string) (*ONNXEmbedder, error) {
	if onnxLibPath != "" {
		ort.SetSharedLibraryPath(onnxLibPath)
	}

	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("initialize ONNX environment: %w", err)
	}

	inputShape := ort.NewShape(1, 512)
	inputTensor, err := ort.NewEmptyTensor[int64](inputShape)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	attentionMaskShape := ort.NewShape(1, 512)
	attentionMaskTensor, err := ort.NewEmptyTensor[int64](attentionMaskShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("create attention mask tensor: %w", err)
	}

	tokenTypeIDsShape := ort.NewShape(1, 512)
	tokenTypeIDsTensor, err := ort.NewEmptyTensor[int64](tokenTypeIDsShape)
	if err != nil {
		inputTensor.Destroy()
		attentionMaskTensor.Destroy()
		return nil, fmt.Errorf("create token type ids tensor: %w", err)
	}

	outputShape := ort.NewShape(1, 512, EmbeddingDim)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		attentionMaskTensor.Destroy()
		tokenTypeIDsTensor.Destroy()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}

	inputNames := []string{"input_ids", "attention_mask", "token_type_ids"}
	outputNames := []string{"last_hidden_state"}

	session, err := ort.NewAdvancedSession(modelPath,
		inputNames, outputNames,
		[]ort.Value{inputTensor, attentionMaskTensor, tokenTypeIDsTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		inputTensor.Destroy()
		attentionMaskTensor.Destroy()
		tokenTypeIDsTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("create ONNX session: %w", err)
	}

	return &ONNXEmbedder{
		session: session,
		inputs:  []ort.Value{inputTensor, attentionMaskTensor, tokenTypeIDsTensor},
		outputs: []ort.Value{outputTensor},
	}, nil
}

func (e *ONNXEmbedder) Embed(inputIDs, attentionMask, tokenTypeIDs []int64) ([]float32, error) {
	inputTensor := e.inputs[0].(*ort.Tensor[int64])
	attentionMaskTensor := e.inputs[1].(*ort.Tensor[int64])
	tokenTypeIDsTensor := e.inputs[2].(*ort.Tensor[int64])

	inputData := inputTensor.GetData()
	attentionMaskData := attentionMaskTensor.GetData()
	tokenTypeIDsData := tokenTypeIDsTensor.GetData()

	for i := range inputData {
		if i < len(inputIDs) {
			inputData[i] = inputIDs[i]
		} else {
			inputData[i] = 0
		}
	}

	for i := range attentionMaskData {
		if i < len(attentionMask) {
			attentionMaskData[i] = attentionMask[i]
		} else {
			attentionMaskData[i] = 0
		}
	}

	for i := range tokenTypeIDsData {
		if i < len(tokenTypeIDs) {
			tokenTypeIDsData[i] = tokenTypeIDs[i]
		} else {
			tokenTypeIDsData[i] = 0
		}
	}

	if err := e.session.Run(); err != nil {
		return nil, fmt.Errorf("run ONNX session: %w", err)
	}

	outputTensor := e.outputs[0].(*ort.Tensor[float32])
	outputData := outputTensor.GetData()

	seqLen := len(attentionMask)
	lastHiddenState := make([][][]float32, 1)
	lastHiddenState[0] = make([][]float32, seqLen)
	for i := 0; i < seqLen; i++ {
		lastHiddenState[0][i] = make([]float32, EmbeddingDim)
		copy(lastHiddenState[0][i], outputData[i*EmbeddingDim:(i+1)*EmbeddingDim])
	}

	return MeanPool(lastHiddenState, attentionMask), nil
}

func (e *ONNXEmbedder) Destroy() {
	for _, v := range e.inputs {
		v.Destroy()
	}
	for _, v := range e.outputs {
		v.Destroy()
	}
	e.session.Destroy()
	ort.DestroyEnvironment()
}
