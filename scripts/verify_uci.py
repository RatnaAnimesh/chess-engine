import subprocess
import time
import sys
import os

def verify_uci():
    print("--- Verifying UCI Protocol ---")
    
    # Path to main.py
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script = os.path.join(project_root, "src", "main.py")
    
    # Launch engine process
    # Run as module from project root
    cmd = [sys.executable, "-m", "src.main"]
    
    print(f"Launching: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1 # Line buffered
    )
    
    def send(msg):
        print(f"> {msg}")
        process.stdin.write(msg + "\n")
        process.stdin.flush()
        
    def read_response(timeout=5):
        start = time.time()
        lines = []
        while time.time() - start < timeout:
            line = process.stdout.readline()
            if line:
                print(f"< {line.strip()}")
                lines.append(line.strip())
                if line.strip() == "uciok" or line.strip() == "readyok" or line.strip().startswith("bestmove"):
                    return lines
            else:
                time.sleep(0.1)
        return lines

    try:
        # 1. Handshake
        send("uci")
        lines = read_response()
        assert "uciok" in lines[-1]
        
        # 2. IsReady
        send("isready")
        lines = read_response()
        assert "readyok" in lines[-1]
        
        # 3. Position + Go
        send("position startpos")
        send("go movetime 1000")
        
        # Wait for bestmove
        lines = read_response(timeout=20) # Give it time to load model and search
        found_bestmove = any(l.startswith("bestmove") for l in lines)
        
        if found_bestmove:
            print("SUCCESS: Engine returned bestmove.")
        else:
            print("FAILURE: Engine did not return bestmove in time.")
            
        # 4. Quit
        send("quit")
        process.wait(timeout=2)
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("STDERR:", process.stderr.read())
    finally:
        if process.poll() is None:
            process.kill()
            
    print("Verification Complete.")

if __name__ == "__main__":
    verify_uci()
