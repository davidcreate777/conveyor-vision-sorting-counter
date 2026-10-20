# counter.py
"""
Object counting module.
Defines the virtual line crossing logic for left-to-right movement.
Maintains counters for cubes and Object B.
"""

import config

class ObjectCounter:
    def __init__(self, line_x=config.COUNTING_LINE_X):
        self.line_x = line_x
        self.counted_ids = set()
        
        # Initialize counts
        self.counts = {
            "Red Cube": 0,
            "Blue Cube": 0,
            "Green Cube": 0,
            "Yellow Cube": 0,
            "Orange Cube": 0,
            "Purple Cube": 0,
            "Unknown Cube": 0,
            "Unknown": 0
        }

    def check_crossings(self, tracker):
        """
        Check if any tracked objects have crossed the virtual line from left to right.
        Updates counters for objects that cross the line for the first time.
        Args:
            tracker: CentroidTracker instance containing history of tracked objects.
        Returns:
            list of dicts containing details of objects that crossed in this frame:
                {'id': obj_id, 'class': class_name}
        """
        crossings = []

        for obj_id, history in tracker.history.items():
            # If the object was already counted, skip
            if obj_id in self.counted_ids:
                continue

            # We need at least 2 points to check crossing
            if len(history) < 2:
                continue

            # Get the last two positions of the object
            prev_pos = history[-2]
            curr_pos = history[-1]

            prev_x, prev_y = prev_pos
            curr_x, curr_y = curr_pos

            # Left-to-right crossing logic:
            # Centroid X was strictly to the left of the line, and is now at or to the right of the line.
            if prev_x < self.line_x <= curr_x:
                # Get the majority class for this object
                obj_class = tracker.get_class(obj_id)
                
                # Update counters
                if obj_class in self.counts:
                    self.counts[obj_class] += 1
                else:
                    self.counts["Unknown"] += 1
                
                self.counted_ids.add(obj_id)
                crossings.append({
                    "id": obj_id,
                    "class": obj_class
                })
                print(f"[Counter] Object {obj_id} crossed the line! Classified as: {obj_class}")

        return crossings

    def get_counts(self):
        """Return a copy of the current counts."""
        return self.counts.copy()

    def reset(self):
        """Reset all counters and tracked IDs."""
        self.counted_ids.clear()
        for key in self.counts:
            self.counts[key] = 0
        print("Counters reset.")
