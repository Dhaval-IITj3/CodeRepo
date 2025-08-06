import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define the notes and their MIDI numbers
notes = [
    ("G3", 55), ("A3", 57), ("B3", 59),
    ("C4", 60), ("D4", 62), ("E4", 64), ("F4", 65), ("G4", 67), ("A4", 69)
]

# Mapping of note names to staff line positions (0 is bottom line of the treble staff)
note_positions = {
    "G3": -2, "A3": -1, "B3": 0,
    "C4": 1, "D4": 2, "E4": 3, "F4": 4, "G4": 5, "A4": 6
}

fig, ax = plt.subplots(figsize=(10, 3))

# Draw the 5 staff lines
for i in range(5):
    ax.plot([0, 9], [i, i], color='black')

# Draw notes as filled circles with MIDI numbers
for i, (note, midi) in enumerate(notes):
    x = i + 0.5
    y = note_positions[note]
    ax.add_patch(mpatches.Circle((x, y), 0.2, color='black'))
    ax.text(x, y + 0.4, f"{note}\n{midi}", ha='center', fontsize=8)

# Aesthetics
ax.set_xlim(0, 9)
ax.set_ylim(-3, 7)
ax.axis('off')
ax.set_title("Treble Staff Notes with MIDI Numbers (G3 to A4)")

plt.tight_layout()
plt.show()
