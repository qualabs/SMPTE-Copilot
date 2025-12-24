"""
Audio processing module for transcribing audio/video files using Whisper.
This module provides functions to transcribe audio and inspect transcription results.
"""

from pathlib import Path
from typing import Optional, Union
import whisper


def transcribe_audio(
    audio_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    save_to_file: bool = True,
    include_timestamps: bool = True,
    return_full_result: bool = False,
) -> Union[str, dict]:
    """
    Transcribe an audio file using Whisper and optionally save to a file.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Optional directory to save the transcription file.
                   If None, saves in the same directory as the audio file.
        save_to_file: If True, saves the transcription to a .txt file
        include_timestamps: If True, includes timestamps in the output
        return_full_result: If True, returns the complete Whisper result dictionary
                          with all available parameters (text, segments, language, etc.)
    
    Returns:
        If return_full_result is True: Complete Whisper result dictionary with keys:
            - 'text' (str): Full transcription
            - 'segments' (list): List of segment dicts with detailed info
            - 'language' (str): Detected language code
        
        If return_full_result is False and include_timestamps is False: 
            Just the transcribed text as a string
        
        If return_full_result is False and include_timestamps is True: 
            Dictionary with 'text' and 'segments' keys only
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load Whisper model and transcribe
    model = whisper.load_model("turbo")
    # Whisper automatically segments based on:
    # - Natural pauses in speech
    # - Silence detection
    # - Semantic boundaries
    # - Speech pattern recognition
    result = model.transcribe(str(audio_path))
    transcribed_text = result["text"]
    segments = result.get("segments", [])
    
    # Save to file if requested
    if save_to_file:
        if output_dir is None:
            output_dir = audio_path.parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output filename (same name as audio file but with .txt extension)
        output_filename = audio_path.stem + "_transcription.txt"
        output_path = output_dir / output_filename
        
        # Write transcription to file
        with open(output_path, 'w', encoding='utf-8') as f:
            if include_timestamps and segments:
                # Write with timestamps
                f.write("=== TRANSCRIPTION WITH TIMESTAMPS ===\n\n")
                for segment in segments:
                    start_time = segment.get("start", 0)
                    end_time = segment.get("end", 0)
                    text = segment.get("text", "").strip()
                    
                    # Format time as HH:MM:SS.mmm
                    start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{start_time%60:06.3f}"
                    end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{end_time%60:06.3f}"
                    
                    f.write(f"[{start_str} --> {end_str}] {text}\n")
                
                f.write("\n\n=== FULL TEXT (NO TIMESTAMPS) ===\n\n")
                f.write(transcribed_text)
            else:
                # Write just the text
                f.write(transcribed_text)
        
        print(f"✓ Transcription saved to: {output_path}")
        if include_timestamps and segments:
            print(f"  - Includes timestamps for {len(segments)} segments")
    
    # Return appropriate format
    if return_full_result:
        # Return complete Whisper result with all parameters
        return result
    elif include_timestamps:
        # Return simplified dict with just text and segments
        return {
            "text": transcribed_text,
            "segments": segments
        }
    else:
        # Return just the text string
        return transcribed_text


def inspect_transcript_result(
    result: dict,
    output_path: Optional[Union[str, Path]] = None,
    audio_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Save the complete Whisper transcript result dictionary to a JSON file.
    
    Args:
        result: The dictionary returned by model.transcribe()
        output_path: Optional path to save the JSON file.
                    If None, saves in the same directory as audio_path or current directory.
        audio_path: Optional audio file path (used to generate output filename if output_path not provided)
    
    Returns:
        Path to the saved JSON file
    """
    import json
    
    # Determine output path
    if output_path is None:
        if audio_path:
            audio_path = Path(audio_path)
            output_path = audio_path.parent / f"{audio_path.stem}_result.json"
        else:
            output_path = Path("transcript_result.json")
    else:
        output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format, handling non-serializable types
    def json_serializer(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, list):
            return [json_serializer(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: json_serializer(v) for k, v in obj.items()}
        else:
            return str(obj)
    
    try:
        json_result = json_serializer(result)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Complete transcript result saved to: {output_path}")
        print(f"  - File size: {output_path.stat().st_size / 1024:.2f} KB")
        print(f"  - Contains {len(result.get('segments', []))} segments")
        
        return output_path
    except Exception as e:
        raise Exception(f"Could not save result to JSON file: {e}")


if __name__ == "__main__":
    # Example usage
    audio_path = Path("/Users/andresmendez/Desktop/Test/videos/10-14 11-00 Session Room AB Li Meininger LED 3D Cinema Display - The Future Display System in Cinema.mp4")
    output_transcription_directory = Path("/Users/andresmendez/Desktop/Test/Transcriptions")
    
    print(f"Transcribing audio file: {audio_path}")
    print("="*60)
    
    try:
        result = transcribe_audio(
            audio_path=audio_path,
            output_dir=output_transcription_directory,
            save_to_file=True,
            return_full_result=True
        )
        
        print(f"\n{'='*60}")
        print("Transcription completed!")
        print(f"Detected language: {result.get('language', 'unknown')}")
        print(f"Number of segments: {len(result.get('segments', []))}")
        print(f"{'='*60}")
        
        # Save the complete result dictionary to a JSON file
        print("\n")
        result_file_path = output_transcription_directory / f"{audio_path.stem}_result.json"
        result_file = inspect_transcript_result(
            result,
            output_path=result_file_path
        )
        
    except Exception as e:
        print(f"❌ Error transcribing audio: {e}")
