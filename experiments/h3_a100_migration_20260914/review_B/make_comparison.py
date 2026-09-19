"""Reuse the existing synchronized video-review renderer for formal A100 B."""
import importlib.util
from pathlib import Path
import av
from PIL import Image

here = Path(__file__).resolve().parent
experiments = here.parents[1]
template = experiments / 'h3_v2v_minimal_20260913/results_31731/b_20260914T045104Z_b52c1cc39548462f9a6b1f6124fef1ea/visual_review/make_side_by_side.py'
spec = importlib.util.spec_from_file_location('review_template', template)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
review.HERE = here
review.SOURCE = here.parent / 'data/shirt_red_couple_124/source.mp4'
review.TARGET = here / 'B_formal_50step.mp4'
review.OUTPUT = here / 'source_vs_B_synchronized.mp4'
review.RECORD = here / 'side_by_side_record.json'
review.EXPECTED = {
    review.SOURCE: '4fb53ba396d3695eb3bb4eadb0b9fe64dc88592e1c8baf3f6cab59a43f66b0a2',
    review.TARGET: 'b80fe4dedf5a10b4edb75ba4d21bc80d19e2718af7693570cf74c2be1170d808',
}
review.main()
contact = Image.new('RGB', (1344, 428 * 4))
with av.open(str(review.OUTPUT)) as video:
    row = 0
    for i, frame in enumerate(video.decode(video=0)):
        if i in (0, 40, 80, 123):
            contact.paste(frame.to_image(), (0, row * 428))
            row += 1
assert row == 4
contact.save(here / 'contact_sheet.png')
