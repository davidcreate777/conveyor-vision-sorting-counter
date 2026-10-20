# tracker.py
"""
Centroid tracker module.
Tracks objects across frames using Euclidean distance matching.
Stores object histories and classification votes to ensure stable labeling.
"""

from collections import OrderedDict
import numpy as np
import config

class CentroidTracker:
    def __init__(self, max_disappeared=config.MAX_DISAPPEARED_FRAMES, max_distance=config.MAX_TRACKING_DISTANCE):
        self.next_object_id = 0
        self.objects = OrderedDict()       # ID -> current centroid (cx, cy)
        self.bboxes = OrderedDict()        # ID -> current bbox (x, y, w, h)
        self.disappeared = OrderedDict()   # ID -> consecutive frames lost
        self.history = OrderedDict()       # ID -> list of recent centroids [(cx, cy), ...]
        self.class_votes = OrderedDict()   # ID -> list of classification results ["Red Cube", ...]
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, initial_class="Unknown"):
        """Register a new object."""
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.bboxes[obj_id] = bbox
        self.disappeared[obj_id] = 0
        self.history[obj_id] = [centroid]
        self.class_votes[obj_id] = [initial_class]
        self.next_object_id += 1
        return obj_id

    def deregister(self, obj_id):
        """Deregister an object."""
        if obj_id in self.objects:
            del self.objects[obj_id]
        if obj_id in self.bboxes:
            del self.bboxes[obj_id]
        if obj_id in self.disappeared:
            del self.disappeared[obj_id]
        if obj_id in self.history:
            del self.history[obj_id]
        if obj_id in self.class_votes:
            del self.class_votes[obj_id]

    def add_class_vote(self, obj_id, class_name):
        """Add a classification vote for the object."""
        if obj_id in self.class_votes:
            self.class_votes[obj_id].append(class_name)
            # Limit history of votes to keep it responsive but stable (e.g. last 15 votes)
            if len(self.class_votes[obj_id]) > 15:
                self.class_votes[obj_id].pop(0)

    def get_class(self, obj_id):
        """Get the majority-vote class for the object."""
        if obj_id not in self.class_votes or not self.class_votes[obj_id]:
            return "Unknown"
        votes = self.class_votes[obj_id]
        # Return the most frequent item
        return max(set(votes), key=votes.count)

    def update(self, rects, classifier=None, frame=None):
        """
        Update the tracker with new detections.
        Args:
            rects: List of dicts from detector.detect() containing 'bbox' and 'centroid'.
            classifier: ColorClassifier instance (optional, for frame-by-frame color voting).
            frame: Cropped frame (optional, for classifier input).
        Returns:
            Dictionary of active objects ID -> current centroid (cx, cy)
        """
        # If there are no detections, increment disappeared count for all active objects
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        # Extract input centroids and bounding boxes
        input_centroids = np.array([r['centroid'] for r in rects], dtype=np.int32)
        input_bboxes = [r['bbox'] for r in rects]

        # If we are not tracking any objects yet, register all detections
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                initial_class = "Unknown"
                if classifier and frame is not None:
                    initial_class = classifier.classify(frame, input_bboxes[i])
                if initial_class == "Object B":
                    continue  # Skip registering white objects
                self.register(input_centroids[i], input_bboxes[i], initial_class)
            return self.objects

        # Grab the set of object IDs and their corresponding centroids
        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()), dtype=np.int32)

        # Compute the distance matrix between object centroids and input centroids
        # D is shape (len(object_centroids), len(input_centroids))
        D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)

        # Find the smallest value in each row and sort the row indexes based on their minimum values
        rows = D.min(axis=1).argsort()

        # Find the smallest value in each column and sort using the row index sorting
        cols = D.argmin(axis=1)[rows]

        # Keep track of which rows and columns we have already examined
        used_rows = set()
        used_cols = set()

        # Loop over the combination of (row, col) index matching
        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # If the distance is too large, don't associate them
            if D[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            
            # Perform color classification and add vote
            if classifier and frame is not None:
                current_class = classifier.classify(frame, input_bboxes[col])
                if current_class == "Object B":
                    # If an existing object is now classified as white, deregister it immediately
                    self.deregister(obj_id)
                    used_rows.add(row)
                    used_cols.add(col)
                    continue
                self.add_class_vote(obj_id, current_class)

            self.objects[obj_id] = input_centroids[col]
            self.bboxes[obj_id] = input_bboxes[col]
            self.disappeared[obj_id] = 0
            
            # Append to position history
            self.history[obj_id].append(tuple(input_centroids[col]))
            if len(self.history[obj_id]) > 30:
                self.history[obj_id].pop(0)

            used_rows.add(row)
            used_cols.add(col)

        # Compute the row and column indexes we have NOT yet examined
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        # Handle disappeared objects
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        # Register new objects
        for col in unused_cols:
            initial_class = "Unknown"
            if classifier and frame is not None:
                initial_class = classifier.classify(frame, input_bboxes[col])
            if initial_class == "Object B":
                continue  # Skip registering white objects
            self.register(input_centroids[col], input_bboxes[col], initial_class)

        return self.objects
