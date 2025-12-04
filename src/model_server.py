import torch
import time
import threading
import queue
from concurrent.futures import Future

class ModelServer:
    def __init__(self, model, device, max_batch_size=16, timeout=0.01):
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.server_thread = threading.Thread(target=self._loop)
        self.server_thread.daemon = True
        
    def start(self):
        self.stop_event.clear()
        self.server_thread.start()
        
    def stop(self):
        self.stop_event.set()
        self.server_thread.join()
        
    def predict(self, state_tensor):
        """
        Submit a prediction request.
        state_tensor: (119, 8, 8) numpy array or tensor
        Returns: Future object that will hold (policy, value)
        """
        future = Future()
        self.queue.put((state_tensor, future))
        return future
        
    def _loop(self):
        while not self.stop_event.is_set():
            batch = []
            futures = []
            
            # 1. Collect batch
            try:
                # Wait for first item
                item = self.queue.get(timeout=0.1)
                batch.append(item[0])
                futures.append(item[1])
                
                # Collect more up to max_batch_size or timeout
                start_wait = time.time()
                while len(batch) < self.max_batch_size:
                    if time.time() - start_wait > self.timeout:
                        break
                    try:
                        item = self.queue.get_nowait()
                        batch.append(item[0])
                        futures.append(item[1])
                    except queue.Empty:
                        time.sleep(0.0001) # Tiny sleep to yield
                        continue
                        
            except queue.Empty:
                continue
                
            # 2. Process batch
            if batch:
                self._process_batch(batch, futures)
                
    def _process_batch(self, batch_states, futures):
        # batch_states: list of (119, 8, 8)
        # Stack them
        if not batch_states:
            return
            
        # Convert to tensor
        # Assuming inputs are numpy arrays or tensors
        import numpy as np
        
        if isinstance(batch_states[0], np.ndarray):
            tensor = torch.from_numpy(np.stack(batch_states))
        else:
            tensor = torch.stack(batch_states)
            
        tensor = tensor.float().to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            policy_logits, values = self.model(tensor)
            
        # Move to CPU
        policy_logits = policy_logits.cpu().numpy()
        values = values.cpu().numpy()
        
        # Distribute results
        for i, future in enumerate(futures):
            future.set_result((policy_logits[i], values[i]))
