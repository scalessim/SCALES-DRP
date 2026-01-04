from astropy.table import Table, unique
import os
import logging


from astropy.table import Table, unique
import os
import logging


class Proctab:
    """
    Lightweight processing table for SCALES quicklook/pipeline products.

    Expected header keywords (best-effort):
      CAMERA   : 'Im' or 'IFS' (we normalize to 'IM'/'IFS')
      IFSMODE  : string (only meaningful if CAMERA=='IFS')
      ACQMODE  : e.g. 'UTR', 'CDS'
      MCLOCK   : e.g. '5.0MHZ'
      IMTYPE   : 'DARK', 'BIAS', 'OBJECT', ...
      EXPTIME  : seconds (float)
      MJD      : float
      TARGNAME : string (or OBJECT as fallback)
    """

    def __init__(self, logger=None):
        self.log = logger if logger is not None else logging.getLogger("SCALES")
        self.proctab = None

    # -------------------------
    # helpers
    # -------------------------
    @staticmethod
    def _as_str(val, default="NONE"):
        if val is None:
            return default
        try:
            s = str(val).strip()
        except Exception:
            return default
        return s if s else default

    @staticmethod
    def _as_float(val, default=float("nan")):
        try:
            if val is None:
                return default
            return float(val)
        except Exception:
            return default

    @staticmethod
    def _norm_camera(cam):
        cam = Proctab._as_str(cam).upper()
        # accept "IM", "Im", "IMG", etc
        if cam.startswith("IM"):
            return "IM"
        if cam.startswith("IFS"):
            return "IFS"
        return cam if cam != "NONE" else "NONE"

    # -------------------------
    # table i/o
    # -------------------------
    def new_proctab(self):
        cnames = (
            "CAMERA", "IFSMODE", "ACQMODE", "MCLOCK",
            "IMTYPE", "EXPTIME", "MJD", "TARGNAME",
            "filename", "SUFF", "STAGE"
        )
        dtypes = (
            "U8", "U32", "U16", "U16",
            "U16", "f8", "f8", "U64",
            "U256", "U32", "i4"
        )
        meta = {"SCALES DRP PROC TABLE": "new table"}
        self.proctab = Table(names=cnames, dtype=dtypes, meta=meta)

        # formatting
        self.proctab["MJD"].format = "15.6f"
        self.proctab["EXPTIME"].format = "10.3f"

    def read_proctab(self, tfil="scales.proc"):
        if os.path.isfile(tfil):
            self.log.info("reading proc table file: %s", tfil)
            self.proctab = Table.read(tfil, format="ascii.fixed_width")
        else:
            self.log.info("proc table file not found: %s (creating new)", tfil)
            self.new_proctab()

        if self.proctab is None or len(self.proctab) == 0:
            self.new_proctab()

        # Ensure expected columns exist (for backward compatibility)
        needed = {"CAMERA","IFSMODE","ACQMODE","MCLOCK","IMTYPE","EXPTIME","MJD","TARGNAME","filename","SUFF","STAGE"}
        missing = [c for c in needed if c not in self.proctab.colnames]
        if missing:
            self.log.warning("Proctab missing columns %s; recreating a fresh table.", missing)
            self.new_proctab()

        self.proctab["MJD"].format = "15.6f"
        if "EXPTIME" in self.proctab.colnames:
            self.proctab["EXPTIME"].format = "10.3f"

    def write_proctab(self, tfil="scales.proc"):
        if self.proctab is None:
            self.log.info("no proc table to write")
            return
        self.proctab.write(filename=tfil, format="ascii.fixed_width", overwrite=True)
        self.log.info("writing proc table file: %s", tfil)

    # -------------------------
    # update / query
    # -------------------------
    def update_proctab(self, frame, suffix="raw", filename="", newtype=None, dedupe_key="filename"):
        """
        Add/update a row for the newly written product.

        Parameters
        ----------
        frame : object with .header (dict-like)
        suffix : str
            pipeline stage suffix label (e.g. 'ql','mbias','mflat','opt_cube',...)
        filename : str
            output FITS filename/path just written
        newtype : str or None
            optional override of IMTYPE
        dedupe_key : str
            column name used to de-duplicate rows. default: 'filename'
        """
        if self.proctab is None:
            self.log.warning("Proctab not initialized; creating new table.")
            self.new_proctab()

        if frame is None:
            self.log.warning("No frame provided; skipping proctab update.")
            return

        hdr = getattr(frame, "header", None)
        if hdr is None:
            self.log.warning("Frame has no header; skipping proctab update.")
            return

        if not filename:
            self.log.error("No filename given; skipping proctab update.")
            return

        # stage mapping
        stages = {
            "ql": 0,
            "mbias": 1,
            "mflat": 4,
            "lensflat": 4,
            "mdark": 4,
            "opt_cube": 6,
            "chi_cube": 7,
        }
        stage = stages.get(suffix, 0)

        # read header values (best effort)
        camera = self._norm_camera(hdr.get("CAMERA", hdr.get("OBSMODE", "NONE")))
        ifsmode = self._as_str(hdr.get("IFSMODE", hdr.get("IFUNAM", "NONE")))
        acqmode = self._as_str(hdr.get("ACQMODE", "NONE"))
        mclock  = self._as_str(hdr.get("MCLOCK", "NONE"))

        imtype  = self._as_str(newtype if newtype is not None else hdr.get("IMTYPE", "NONE"))
        exptime = self._as_float(hdr.get("EXPTIME", hdr.get("TTIME", None)), default=float("nan"))
        mjd     = self._as_float(hdr.get("MJD", None), default=float("nan"))

        targname = self._as_str(hdr.get("TARGNAME", hdr.get("OBJECT", "NONE")))
        targname = targname.replace(" ", "") if targname != "NONE" else "NONE"

        # rules you requested
        if camera == "IM":
            ifsmode = "NONE"
        elif camera == "IFS":
            ifsmode = self._as_str(ifsmode, default="NONE")
        else:
            # unknown camera; still keep something consistent
            if camera == "NONE":
                camera = "NONE"
            ifsmode = self._as_str(ifsmode, default="NONE")

        new_row = [
            camera, ifsmode, acqmode, mclock,
            imtype, exptime, mjd, targname,
            self._as_str(filename), self._as_str(suffix), int(stage)
        ]

        self.proctab.add_row(new_row)

        # de-duplicate
        if dedupe_key in self.proctab.colnames:
            self.proctab = unique(self.proctab, keys=[dedupe_key], keep="last")

        # sort (use MJD if present)
        if "MJD" in self.proctab.colnames:
            try:
                self.proctab.sort("MJD")
            except Exception:
                pass

        self.log.info("proctab updated: SUFF=%s file=%s", suffix, filename)


    def search_proctab(self, frame, target_type=None, nearest=False):
        """
        Find rows matching CAMERA (+ optional IMTYPE), optionally nearest in MJD.
        """
        if self.proctab is None or len(self.proctab) == 0:
            self.log.warning("Proctab is empty")
            return None
        if frame is None or getattr(frame, "header", None) is None:
            self.log.warning("No frame/header for proctab search")
            return None

        hdr = frame.header
        camera = self._norm_camera(hdr.get("CAMERA", hdr.get("OBSMODE", "NONE")))

        tab = self.proctab[self.proctab["CAMERA"] == camera]

        if target_type is not None:
            tab = tab[tab["IMTYPE"] == self._as_str(target_type).upper()]

        if nearest and len(tab) > 1 and "MJD" in tab.colnames:
            tf = self._as_float(hdr.get("MJD", None), default=None)
            if tf is not None:
                # pick closest MJD
                dm = [abs(row["MJD"] - tf) for row in tab]
                j = int(min(range(len(dm)), key=lambda k: dm[k]))
                tab = tab[j:j+1]

        return tab

    def in_proctab(self, frame):
        """
        Check if frame MJD exists for that CAMERA in table.
        """
        if self.proctab is None or frame is None or getattr(frame, "header", None) is None:
            return False
        camera = self._norm_camera(frame.header.get("CAMERA", frame.header.get("OBSMODE", "NONE")))
        mjd = self._as_float(frame.header.get("MJD", None), default=None)
        if mjd is None:
            return False
        tab = self.proctab[self.proctab["CAMERA"] == camera]
        return mjd in set(tab["MJD"])
