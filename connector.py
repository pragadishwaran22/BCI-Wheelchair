import socket
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('connector.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def send_command_to_unity(command, unity_ip="127.0.0.1", port=5055):
    """
    Send a command to Unity application via UDP socket
    
    Args:
        command (str): Command to send (e.g., "LEFT", "RIGHT", "FOOT", "TONGUE")
        unity_ip (str): IP address of Unity application (default: localhost)
        port (int): Port number Unity is listening on (default: 5055)
    
    Returns:
        bool: True if command sent successfully, False otherwise
    """
    try:
        logger.info(f"Initializing UDP socket connection...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        logger.info(f"Preparing to send command: '{command}' to {unity_ip}:{port}")
        
        # Send the command
        sock.sendto(command.encode(), (unity_ip, port))
        
        logger.info(f"Command '{command}' sent successfully to Unity at {unity_ip}:{port}")
        
        # Close the socket
        sock.close()
        logger.info("Socket connection closed")
        
        return True
        
    except socket.error as e:
        logger.error(f"Socket error occurred: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    logger.info("=== BCI Unity Connector Started ===")
        
    # Configuration
    unity_ip = "127.0.0.1"
    port = 5055
    command = "LEFT"
    
    logger.info(f"Configuration - Unity IP: {unity_ip}, Port: {port}")
    
    # Send command
    success = send_command_to_unity(command, unity_ip, port)
    
    if success:
        logger.info("=== BCI Unity Connector Completed Successfully ===")
    else:
        logger.error("=== BCI Unity Connector Failed ===")
        sys.exit(1)