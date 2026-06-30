import cv2
import time
import threading
from ultralytics import YOLO
from flask import Flask, Response, jsonify, request, render_template
import os
from datetime import datetime
from pymongo import MongoClient
from bson import json_util
import asyncio
import winrt.runtime
import winrt.windows.foundation
import winrt.windows.devices.geolocation as wdg
from functools import wraps
from io import StringIO
import csv

app = Flask(__name__)

# Decorator to run async functions in threads
def async_to_sync(async_func):
    @wraps(async_func)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

class RoadDefectDetector:
    def __init__(self):
        # MongoDB setup
        self.client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        try:
            self.client.server_info()
            print("✅ MongoDB connection successful")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            raise

        self.db = self.client["road_defects_db"]
        self.defects_collection = self.db["defects"]

        if "defects" not in self.db.list_collection_names():
            print("Creating 'defects' collection")
            self.db.create_collection("defects")

        model_path = r"C:\Users\USER\Downloads\runs\detect\yolov8_road_defect_\weights\best.pt"
        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        print("✅ Camera initialized successfully")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        self.detection_active = False
        self.detection_thread = None

        self.latest_frame = None
        self.latest_annotated = None
        self.latest_defects = []

        self.frame_count = 0
        self.current_fps = 0
        self.last_fps_update = time.time()

        # GPS initialization
        self.gps_coords = None  # Use None instead of (0,0) to indicate no data
        self.last_gps_update = 0
        self.gps_error = "Initializing GPS..."
        self.use_real_gps = True  # Set to False for simulation
        self.simulated_gps = (37.7749, -122.4194)  # San Francisco coordinates
        
        print("System initialized successfully")

        self.gps_thread = threading.Thread(target=self.update_gps, daemon=True)
        self.gps_thread.start()
        print("📡 GPS thread started")

    @async_to_sync
    async def get_gps_location(self):
        """Async method to get GPS location"""
        try:
            # 1. Request access from Windows first
            access_status = await wdg.Geolocator.request_access_async()
            
            # 2. Check if user/system allowed it
            if access_status != wdg.GeolocationAccessStatus.ALLOWED:
                return None, "Location access denied by Windows"

            locator = wdg.Geolocator()
            # 3. Use default accuracy (HIGH can cause timeouts on desktop PCs without GPS chips)
            locator.desired_accuracy = wdg.PositionAccuracy.DEFAULT
            
            # 4. Fetch position
            pos = await asyncio.wait_for(locator.get_geoposition_async(), timeout=10)
            return (
                pos.coordinate.point.position.latitude,
                pos.coordinate.point.position.longitude
            ), None
        except asyncio.TimeoutError:
            return None, "GPS timeout (No hardware signal)"
        except Exception as e:
            return None, str(e)

    def update_gps(self):
        """Thread function to update GPS coordinates"""
        # Properly initialize the background thread as Multi-Threaded (MTA)
        winrt.runtime.init_apartment(winrt.runtime.ApartmentType.MULTI_THREADED)
        
        while True:
            try:
                if self.use_real_gps:
                    coords, error = self.get_gps_location()
                    if coords:
                        self.gps_coords = coords
                        self.last_gps_update = time.time()
                        self.gps_error = None
                        print(f"📍 Real GPS Coordinates: {self.gps_coords}")
                    else:
                        self.gps_error = error
                        print(f"⚠️ GPS Error: {error}")
                else:
                    # Use simulated GPS
                    self.gps_coords = self.simulated_gps
                    self.last_gps_update = time.time()
                    self.gps_error = None
                    print(f"📍 Simulated GPS: {self.gps_coords}")
                
            except Exception as e:
                self.gps_error = str(e)
                print(f"⚠️ GPS Update Exception: {e}")
            
            time.sleep(5)  # Update every 5 seconds

    def run_detection(self):
        print("🚀 Detection started")
        try:
            while self.detection_active:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ Error reading frame")
                    time.sleep(0.1)
                    continue

                self.latest_frame = frame.copy()

                results = self.model.predict(
                    source=frame,
                    conf=0.3,
                    imgsz=640,
                    device='0',
                    verbose=False
                )

                defects = []
                if results and results[0].boxes:
                    print(f"🔍 Found {len(results[0].boxes)} potential defects")
                    defects = self.process_defects(results)
                else:
                    print("👀 No defects detected in frame")

                self.calculate_fps()

                if results:
                    annotated_frame = results[0].plot()
                else:
                    annotated_frame = frame.copy()

                cv2.putText(annotated_frame, f"FPS: {self.current_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                self.latest_annotated = annotated_frame
                self.latest_defects = defects

        except Exception as e:
            print(f"🔥 Detection error: {e}")
        finally:
            self.cap.release()
            print("📷 Camera released")

    def process_defects(self, results):
        defects = []
        # Get current timestamp once for all defects in this frame
        timestamp = datetime.now()
        
        # Get GPS status
        if self.gps_coords:
            lat, lon = self.gps_coords
            gps_status = "Valid"
        else:
            lat, lon = 0.0, 0.0
            gps_status = f"Invalid: {self.gps_error or 'No GPS data'}"
            print(f"⚠️ Using fallback GPS: {gps_status}")
        
        for i, result in enumerate(results):
            if not result.boxes:
                continue

            for j, box in enumerate(result.boxes):
                conf = float(box.conf)
                if conf > 0.3:
                    class_idx = int(box.cls)
                    class_name = self.model.names.get(class_idx, f"unknown_class_{class_idx}")

                    defect = {
                        'timestamp': timestamp,
                        'class': class_name,
                        'confidence': conf,
                        'latitude': lat,
                        'longitude': lon,
                        'gps_status': gps_status
                    }
                    defects.append(defect)

                    try:
                        result = self.defects_collection.insert_one(defect)
                        print(f"💽 Inserted defect with ID: {result.inserted_id}")
                    except Exception as e:
                        print(f"🚨 Error saving to MongoDB: {e}")
        return defects

    def calculate_fps(self):
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_update
        if elapsed > 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time

    def start(self):
        if not self.detection_active:
            print("🟢 Starting detection system...")
            self.detection_active = True
            self.detection_thread = threading.Thread(target=self.run_detection, daemon=True)
            self.detection_thread.start()

    def stop(self):
        if self.detection_active:
            print("🔴 Stopping detection system...")
            self.detection_active = False
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=2.0)
            print("🛑 Detection system stopped")


detector = RoadDefectDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if detector.latest_annotated is not None:
                ret, jpeg = cv2.imencode('.jpg', detector.latest_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.033)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    detector.start()
    return jsonify({"status": "Detection started", "success": True})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    detector.stop()
    return jsonify({"status": "Detection stopped", "success": True})

@app.route('/get_stats', methods=['GET'])
def get_stats():
    try:
        defect_count = detector.defects_collection.count_documents({})
    except Exception as e:
        print(f"Error getting defect count: {e}")
        defect_count = 0
        
    # Get GPS status for response
    if detector.gps_coords:
        gps_coords = detector.gps_coords
        gps_status = "Active"
    else:
        gps_coords = [0.0, 0.0]
        gps_status = f"Error: {detector.gps_error}"
    
    return jsonify({
        "fps": detector.current_fps,
        "defect_count": defect_count,
        "gps": gps_coords,
        "gps_status": gps_status,
        "last_update": detector.last_gps_update
    })

@app.route('/get_defects')
def get_defects():
    try:
        defects = list(detector.defects_collection.find().sort("timestamp", -1))
        
        # Convert ObjectId to string and format timestamp
        formatted_defects = []
        for defect in defects:
            defect['_id'] = str(defect['_id'])
            defect['timestamp'] = defect['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            formatted_defects.append(defect)
            
        return jsonify({
            "success": True,
            "defects": formatted_defects,
            "count": len(formatted_defects)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Failed to retrieve defects"
        }), 500

@app.route('/test_db')
def test_db():
    try:
        count = detector.defects_collection.count_documents({})
        return f"✅ MongoDB connection working! Defects in DB: {count}"
    except Exception as e:
        return f"❌ MongoDB error: {str(e)}"

@app.route('/get_all_defects', methods=['GET'])
def get_all_defects():
    try:
        # Get all defects sorted by timestamp (newest first)
        defects = list(detector.defects_collection.find().sort("timestamp", -1))
        
        # Format the defects for better readability
        formatted_defects = []
        for defect in defects:
            formatted_defects.append({
                "id": str(defect["_id"]),
                "timestamp": defect["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "type": defect["class"],
                "confidence": f"{defect['confidence']:.2f}",
                "location": {
                    "latitude": defect["latitude"],
                    "longitude": defect["longitude"],
                    "status": defect.get("gps_status", "unknown")
                }
            })
        
        return jsonify({
            "success": True,
            "defects": formatted_defects,
            "count": len(formatted_defects)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Failed to retrieve defects"
        }), 500

@app.route('/api/defect_type_report')
def defect_type_report():
    try:
        pipeline = [
            {
                "$match": {
                    "class": {"$exists": True}
                }
            },
            {
                "$group": {
                    "_id": "$class",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence"},
                    "first_detected": {"$min": "$timestamp"},
                    "last_detected": {"$max": "$timestamp"}
                }
            },
            {
                "$project": {
                    "type": "$_id",
                    "count": 1,
                    "avg_confidence": {"$round": ["$avg_confidence", 2]},
                    "first_detected": 1,
                    "last_detected": 1,
                    "_id": 0
                }
            },
            {"$sort": {"count": -1}}
        ]
        print("Running pipeline:", pipeline)  # Log the pipeline
        results = list(detector.defects_collection.aggregate(pipeline))
        print("Pipeline results:", results)  # Log the results
        
        if not results:
            return jsonify({"success": False, "error": "No data found"})
        
        results = list(detector.defects_collection.aggregate(pipeline))
        
        # Format timestamps
        for item in results:
            item['first_detected'] = item['first_detected'].strftime("%Y-%m-%d %H:%M")
            item['last_detected'] = item['last_detected'].strftime("%Y-%m-%d %H:%M")
            item['avg_confidence'] = f"{item['avg_confidence']*100:.1f}%"
        
        return jsonify({
            "success": True,
            "report": results,
            "total_defects": sum(item['count'] for item in results)
        })
        
    except Exception as e:
        print("🚨 Error in defect_type_report:", str(e))  # Log the full error
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/location_report')
def location_report():
    try:
        # Get all defects with valid GPS coordinates
        valid_defects = list(detector.defects_collection.find({
            "latitude": {"$exists": True, "$ne": None, "$ne": 0},
            "longitude": {"$exists": True, "$ne": None, "$ne": 0}
        }))
        
        # If no defects with valid GPS, return empty response
        if not valid_defects:
            return jsonify({
                "success": True,
                "stats": {
                    "total_defects": 0,
                    "valid_gps": 0,
                    "gps_coverage": "0%",
                    "hotspot_count": 0
                },
                "hotspots": []
            })
        
        # Cluster defects by rounded coordinates (simple clustering)
        clusters = {}
        for defect in valid_defects:
            # Round to 4 decimal places (~11m precision)
            lat_key = round(defect['latitude'], 4)
            lon_key = round(defect['longitude'], 4)
            cluster_key = f"{lat_key}_{lon_key}"
            
            if cluster_key not in clusters:
                clusters[cluster_key] = {
                    "latitude": lat_key,
                    "longitude": lon_key,
                    "defect_count": 0,
                    "types": {},
                    "last_detected": None
                }
            
            clusters[cluster_key]['defect_count'] += 1
            
            # Track defect types
            defect_type = defect.get('class', 'unknown')
            clusters[cluster_key]['types'][defect_type] = clusters[cluster_key]['types'].get(defect_type, 0) + 1
            
            # Track most recent detection
            defect_time = defect['timestamp']
            if (clusters[cluster_key]['last_detected'] is None or 
                defect_time > clusters[cluster_key]['last_detected']):
                clusters[cluster_key]['last_detected'] = defect_time
        
        # Convert to list and format
        hotspots = []
        for cluster in clusters.values():
            # Get most common type
            if cluster['types']:
                main_type = max(cluster['types'].items(), key=lambda x: x[1])[0]
            else:
                main_type = "unknown"
                
            hotspots.append({
                "latitude": cluster['latitude'],
                "longitude": cluster['longitude'],
                "defect_count": cluster['defect_count'],
                "main_type": main_type,
                "last_detected": cluster['last_detected'].strftime("%Y-%m-%d %H:%M")
            })
        
        # Sort by defect count (descending)
        hotspots.sort(key=lambda x: x['defect_count'], reverse=True)
        
        # Get total defect count
        total_defects = detector.defects_collection.count_documents({})
        
        return jsonify({
            "success": True,
            "stats": {
                "total_defects": total_defects,
                "valid_gps": len(valid_defects),
                "gps_coverage": f"{len(valid_defects)/total_defects*100:.1f}%" if total_defects > 0 else "0%",
                "hotspot_count": len(hotspots)
            },
            "hotspots": hotspots[:50]  # Return top 50 hotspots
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
@app.route('/get_defects_years')
def get_defects_years():
    """Get distinct years from defects collection"""
    try:
        pipeline = [
            {
                "$group": {
                    "_id": {"$year": "$timestamp"},
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        years = [str(year['_id']) for year in detector.defects_collection.aggregate(pipeline)]
        return jsonify({"success": True, "years": years})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/get_defects_months')
def get_defects_months():
    """Get distinct months for a specific year"""
    try:
        year = int(request.args.get('year'))
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        
        pipeline = [
            {
                "$match": {
                    "timestamp": {
                        "$gte": start_date,
                        "$lt": end_date
                    }
                }
            },
            {
                "$group": {
                    "_id": {"$month": "$timestamp"},
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        months = [str(month['_id']) for month in detector.defects_collection.aggregate(pipeline)]
        return jsonify({"success": True, "months": months})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/export_defects_csv')
def export_defects_csv():
    """Export defects to CSV with optional year/month filtering"""
    try:
        # Get filter parameters
        year = request.args.get('year', 'all')
        month = request.args.get('month', 'all')
        
        # Build query based on filters
        query = {}
        if year != 'all':
            year = int(year)
            start_date = datetime(year, 1, 1)
            end_date = datetime(year + 1, 1, 1)
            query['timestamp'] = {'$gte': start_date, '$lt': end_date}
            
            if month != 'all':
                month = int(month)
                start_date = datetime(year, month, 1)
                if month == 12:
                    end_date = datetime(year + 1, 1, 1)
                else:
                    end_date = datetime(year, month + 1, 1)
                query['timestamp'] = {'$gte': start_date, '$lt': end_date}
        
        # Get filtered defects
        defects = list(detector.defects_collection.find(query).sort("timestamp", -1))
        
        # Generate filename
        filename = "road_defects"
        if year != 'all':
            filename += f"_{year}"
            if month != 'all':
                filename += f"_{month:02d}"
        else:
            filename += "_all"
        filename += ".csv"
        
        # Create CSV in memory
        si = StringIO()
        cw = csv.writer(si)
        
        # Write header
        cw.writerow([
            'ID', 'Timestamp', 'Defect Type', 'Confidence', 
            'Latitude', 'Longitude', 'GPS Status'
        ])
        
        # Write data rows
        for defect in defects:
            cw.writerow([
                str(defect['_id']),
                defect['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                defect['class'],
                f"{defect['confidence']:.4f}",
                defect['latitude'],
                defect['longitude'],
                defect.get('gps_status', 'unknown')
            ])
        
        # Create response with CSV data
        output = si.getvalue()
        response = Response(
            output,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
        
        return response
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Failed to generate CSV export"
        }), 500
    

if __name__ == '__main__':
    print("🚀 Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, threaded=True)