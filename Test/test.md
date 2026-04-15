flowchart LR
    subgraph S1["⚡ ШАГ 1: Захват с дрона"]
        A1["Газоанализатор<br/>ppm + GPS"]
        A2["OGI-камера<br/>IR-видео"]
        A3["RGB-камера<br/>видео"]
    end

    subgraph S2["🔧 ШАГ 2: Предобработка"]
        B1["ppm_scaled"]
        B2["thermal_frame<br/>224×224×1"]
        B3["rgb_frame<br/>224×224×3"]
    end

    subgraph S3["📊 ШАГ 3: Feature Engineering"]
        C1["methane_features<br/>N×3"]
        C2["thermal_tensor"]
        C3["rgb_tensor"]
    end

    subgraph S4["🧠 ШАГ 4: Обучение"]
        D1["p_methane<br/>0/1"]
        D2["p_thermal<br/>0..1"]
        D3["p_rgb<br/>0..1"]
    end

    subgraph S5["⚖️ ШАГ 5: Fusion"]
        E1["p_leak<br/>0..1"]
    end

    subgraph S6["📈 ШАГ 6: Оценка"]
        F1{"Precision > 0.88?<br/>Recall > 0.92?<br/>FAR ≤ 1?"}
    end

    subgraph S7["💾 ШАГ 7: Экспорт"]
        G1["ONNX / TensorRT"]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3
    
    B1 --> C1
    B2 --> C2
    B3 --> C3
    
    C1 --> D1
    C2 --> D2
    C3 --> D3
    
    D1 --> E1
    D2 --> E1
    D3 --> E1
    
    E1 --> F1
    F1 -->|НЕТ| B1
    F1 -->|ДА| G1
