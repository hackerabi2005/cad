import React from 'react';
import { AlertTriangle, Shield, ExternalLink, Heart } from 'lucide-react';

export function DisclaimerHeaderBanner() {
  return (
    <div
      id="safety-disclaimer-banner"
      className="w-full bg-gradient-to-r from-amber-500/15 via-amber-500/25 to-amber-500/15 border-b border-amber-500/30 px-4 py-2 flex items-center justify-center text-center text-xs font-semibold text-amber-200 select-none shadow-sm z-30"
    >
      <div className="flex items-center gap-2 max-w-5xl">
        <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 animate-pulse" />
        <span>
          <strong>Clinical Safety Disclaimer:</strong> Decision support & educational use only — not a substitute for formal diagnostic imaging, catheterization, or physician judgment.
        </span>
      </div>
    </div>
  );
}

export function DisclaimerFooter() {
  return (
    <footer
      id="safety-disclaimer-footer"
      className="w-full border-t border-slate-800/80 bg-slate-950/90 text-slate-400 text-xs py-4 px-6 mt-8"
    >
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-center md:text-left">
        <div className="space-y-1">
          <p className="font-medium text-slate-300">
            Cardio3D AI — Multimodal Cardiovascular Risk & 3D Coronary Stenosis Viewer
          </p>
          <p className="text-[11px] text-slate-500">
            <strong>Clinical Safety Notice:</strong> Model predictions reflect statistical risk probabilities based on cohort features. Never make diagnostic or intervention decisions without certified angiographic imaging.
          </p>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-4 text-[11px] text-slate-400">
          <span className="flex items-center gap-1">
            <span>Dataset: UCI Extension of Z-Alizadeh Sani</span>
            <span className="text-slate-600">(CC BY 4.0)</span>
          </span>
          <span className="text-slate-700">|</span>
          <span className="flex items-center gap-1">
            <span>3D Mesh: BodyParts3D © DBCLS</span>
            <span className="text-slate-600">(CC BY-SA 2.1 JP)</span>
          </span>
        </div>
      </div>
    </footer>
  );
}
