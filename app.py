from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from werkzeug.utils import secure_filename
from src.data_loader import load_csv_file, load_gdf_epochs, BCI2aDataset
from src.preprocess import normalize_epoch
from src.classifier import SimpleEEGNet1D
import socket
import logging
import threading
import time
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CLASS_NAMES = ['left', 'right', 'foot', 'tongue']
MODEL_PATH = 'models/model_a_lr5e-4_wd5e-4.pth'

# Unity Integration Configuration
UNITY_IP = "127.0.0.1"
UNITY_PORT = 5055
UNITY_ENABLED = True

# Configure logging for Unity integration
logging.basicConfig(level=logging.INFO)
unity_logger = logging.getLogger('unity_connector')

def get_class_name(idx):
    return CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"

def send_command_to_unity(command, unity_ip=UNITY_IP, port=UNITY_PORT):
    """
    Send a command to Unity application via UDP socket
    
    Args:
        command (str): Command to send (e.g., "LEFT", "RIGHT", "FOOT", "TONGUE")
        unity_ip (str): IP address of Unity application
        port (int): Port number Unity is listening on
    
    Returns:
        bool: True if command sent successfully, False otherwise
    """
    if not UNITY_ENABLED:
        unity_logger.info(f"Unity integration disabled. Would have sent: {command}")
        return True
        
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(command.upper().encode(), (unity_ip, port))
        sock.close()
        
        unity_logger.info(f"Command '{command.upper()}' sent to Unity at {unity_ip}:{port}")
        return True
        
    except socket.error as e:
        unity_logger.error(f"Failed to send command to Unity: {e}")
        return False
    except Exception as e:
        unity_logger.error(f"Unexpected error sending to Unity: {e}")
        return False

def send_predictions_to_unity(predictions, probabilities, confidence_threshold=0.7):
    """
    Send BCI predictions to Unity with confidence filtering
    
    Args:
        predictions (np.array): Array of prediction indices
        probabilities (np.array): Array of prediction probabilities
        confidence_threshold (float): Minimum confidence to send command
    
    Returns:
        dict: Summary of commands sent to Unity
    """
    commands_sent = []
    
    for i, (pred_idx, prob_array) in enumerate(zip(predictions, probabilities)):
        confidence = prob_array[pred_idx]
        command = get_class_name(pred_idx)
        
        if confidence >= confidence_threshold:
            success = send_command_to_unity(command)
            commands_sent.append({
                'epoch': i + 1,
                'command': command.upper(),
                'confidence': float(confidence),
                'sent_successfully': success,
                'timestamp': datetime.now().isoformat()
            })
            
            # Add small delay between commands to avoid overwhelming Unity
            time.sleep(0.1)
        else:
            unity_logger.info(f"Skipping command '{command}' (confidence: {confidence:.3f} < {confidence_threshold})")
    
    return {
        'total_predictions': len(predictions),
        'commands_sent': len(commands_sent),
        'confidence_threshold': confidence_threshold,
        'commands': commands_sent
    }

def classify_and_send_to_unity(filepath, is_gdf=False, temperature=1.0, use_ensemble=False, 
                              send_to_unity=False, confidence_threshold=0.7):
    """
    Enhanced classification function that optionally sends predictions to Unity
    
    Args:
        filepath (str): Path to the EEG data file
        is_gdf (bool): Whether the file is in GDF format
        temperature (float): Temperature scaling for predictions
        use_ensemble (bool): Whether to use ensemble of models
        send_to_unity (bool): Whether to send predictions to Unity
        confidence_threshold (float): Minimum confidence to send to Unity
    
    Returns:
        dict: Classification results with optional Unity integration info
    """
    # Get standard classification results
    result = classify_file(filepath, is_gdf, temperature, use_ensemble)
    
    # Add Unity integration if requested
    if send_to_unity and UNITY_ENABLED:
        unity_logger.info("Sending predictions to Unity...")
        
        # Extract predictions and probabilities from results
        predictions = np.array([r['prediction_idx'] for r in result['results']])
        probabilities = np.array([[0.0] * len(CLASS_NAMES) for _ in range(len(predictions))])
        
        # Reconstruct probability arrays from confidence values
        for i, r in enumerate(result['results']):
            probabilities[i][r['prediction_idx']] = r['confidence']
        
        # Send to Unity
        unity_result = send_predictions_to_unity(predictions, probabilities, confidence_threshold)
        result['unity_integration'] = unity_result
        
        unity_logger.info(f"Unity integration complete. Sent {unity_result['commands_sent']} commands.")
    
    return result

def classify_file(filepath, is_gdf=False, temperature=1.0, use_ensemble=False):
    # Load data
    if is_gdf:
        epochs, labels = load_gdf_epochs(filepath, event_id={1:0, 2:1, 3:2, 4:3})
    else:
        epochs, labels = load_csv_file(filepath, seq_len=1000, overlap=700)
    
    # Preprocess
    X = np.array([normalize_epoch(e) for e in epochs]).astype('float32')
    y_true = np.array(labels)
    
    # Load model(s)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    num_classes = checkpoint['classifier.3.bias'].shape[0] if 'classifier.3.bias' in checkpoint else len(np.unique(y_true))
    
    models = []
    if use_ensemble:
        # Try to load multiple models for ensemble
        model_paths = [
            'models/model_a_lr5e-4_wd5e-4.pth',
            'models/model_b_lr3e-4_wd1e-3.pth',
            'models/model_c_lr7e-4_wd2e-3.pth'
        ]
        for mp in model_paths:
            if os.path.exists(mp):
                try:
                    m = SimpleEEGNet1D(in_channels=X.shape[1], n_classes=num_classes, dropout=0.5)
                    m.load_state_dict(torch.load(mp, map_location=device))
                    m.to(device).eval()
                    models.append(m)
                except:
                    pass
    
    # If no ensemble models or ensemble failed, use single model
    if len(models) == 0:
        model = SimpleEEGNet1D(in_channels=X.shape[1], n_classes=num_classes, dropout=0.0)  # No dropout during inference
        model.load_state_dict(checkpoint)
        model.to(device).eval()
        # Ensure dropout is disabled
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()
        models = [model]
    else:
        # Ensure dropout is disabled for all ensemble models
        for model in models:
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.eval()
    
    # Classify with ensemble or single model
    loader = DataLoader(BCI2aDataset(X, y_true), batch_size=64, shuffle=False)
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            
            # Ensemble: average logits from multiple models
            if len(models) > 1:
                logits_sum = None
                for m in models:
                    logits = m(xb)
                    if logits_sum is None:
                        logits_sum = logits
                    else:
                        logits_sum = logits_sum + logits
                logits = logits_sum / len(models)
            else:
                logits = models[0](xb)
            
            # Apply temperature scaling for better calibration
            logits_scaled = logits / temperature
            probs = torch.softmax(logits_scaled, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
            probabilities.extend(probs.cpu().numpy())
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Calculate metrics
    accuracy = (predictions == y_true).mean() if len(y_true) > 0 else 0.0
    correct = (predictions == y_true).sum() if len(y_true) > 0 else 0
    
    # Per-class metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    if len(y_true) > 0 and len(np.unique(y_true)) > 1:
        precision = precision_score(y_true, predictions, average='macro', zero_division=0)
        recall = recall_score(y_true, predictions, average='macro', zero_division=0)
        f1 = f1_score(y_true, predictions, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, predictions).tolist()
    else:
        precision = recall = f1 = 0.0
        cm = []
    
    # Prepare results
    results = []
    for i in range(len(predictions)):
        results.append({
            'row': i + 1,
            'prediction': get_class_name(predictions[i]),
            'prediction_idx': int(predictions[i]),
            'confidence': float(probabilities[i][predictions[i]]),
            'true_label': get_class_name(y_true[i]) if len(y_true) > 0 else None,
            'correct': bool(predictions[i] == y_true[i]) if len(y_true) > 0 else None
        })
    
    return {
        'accuracy': float(accuracy),
        'correct': int(correct),
        'total': len(predictions),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm,
        'results': results
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get parameters
    temperature = float(request.form.get('temperature', 1.0))
    use_ensemble = request.form.get('ensemble', 'false').lower() == 'true'
    send_to_unity = request.form.get('send_to_unity', 'false').lower() == 'true'
    confidence_threshold = float(request.form.get('confidence_threshold', 0.7))
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Determine file type
        is_gdf = filename.lower().endswith('.gdf')
        
        # Classify with optional Unity integration
        result = classify_and_send_to_unity(
            filepath, 
            is_gdf=is_gdf, 
            temperature=temperature, 
            use_ensemble=use_ensemble,
            send_to_unity=send_to_unity,
            confidence_threshold=confidence_threshold
        )
        
        # Cleanup
        os.remove(filepath)
        
        return jsonify(result)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/unity/send_command', methods=['POST'])
def unity_send_command():
    """
    Manually send a command to Unity
    """
    data = request.get_json()
    if not data or 'command' not in data:
        return jsonify({'error': 'Command is required'}), 400
    
    command = data['command'].upper()
    if command not in [name.upper() for name in CLASS_NAMES]:
        return jsonify({'error': f'Invalid command. Must be one of: {CLASS_NAMES}'}), 400
    
    success = send_command_to_unity(command)
    
    return jsonify({
        'success': success,
        'command': command,
        'timestamp': datetime.now().isoformat(),
        'unity_enabled': UNITY_ENABLED
    })

@app.route('/unity/status', methods=['GET'])
def unity_status():
    """
    Get Unity integration status
    """
    return jsonify({
        'unity_enabled': UNITY_ENABLED,
        'unity_ip': UNITY_IP,
        'unity_port': UNITY_PORT,
        'supported_commands': CLASS_NAMES,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/unity/toggle', methods=['POST'])
def unity_toggle():
    """
    Toggle Unity integration on/off
    """
    global UNITY_ENABLED
    data = request.get_json()
    
    if data and 'enabled' in data:
        UNITY_ENABLED = bool(data['enabled'])
    else:
        UNITY_ENABLED = not UNITY_ENABLED
    
    unity_logger.info(f"Unity integration {'enabled' if UNITY_ENABLED else 'disabled'}")
    
    return jsonify({
        'unity_enabled': UNITY_ENABLED,
        'message': f"Unity integration {'enabled' if UNITY_ENABLED else 'disabled'}",
        'timestamp': datetime.now().isoformat()
    })

@app.route('/unity/test_connection', methods=['POST'])
def unity_test_connection():
    """
    Test connection to Unity by sending a test command
    """
    test_command = "LEFT"  # Default test command
    data = request.get_json()
    if data and 'command' in data:
        test_command = data['command'].upper()
    
    if test_command not in [name.upper() for name in CLASS_NAMES]:
        return jsonify({'error': f'Invalid test command. Must be one of: {CLASS_NAMES}'}), 400
    
    success = send_command_to_unity(test_command)
    
    return jsonify({
        'connection_successful': success,
        'test_command': test_command,
        'unity_ip': UNITY_IP,
        'unity_port': UNITY_PORT,
        'unity_enabled': UNITY_ENABLED,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

