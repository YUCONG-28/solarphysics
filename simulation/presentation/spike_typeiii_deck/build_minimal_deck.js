'use strict';

const path = require('path');

const userProfile = process.env.USERPROFILE;
if (!userProfile) {
  throw new Error('USERPROFILE is required to locate presentation-skill.');
}

const skillRoot = path.join(userProfile, '.codex', 'skills', 'presentation-skill');
const slidesPath = path.join(skillRoot, 'templates', 'pptxgenjs', 'slides.js');
const builderPath = path.join(skillRoot, 'scripts', 'build_deck_pptxgenjs.js');
const slides = require(slidesPath);

slides.renderTitle = function renderMinimalTitle(_pptx, slide, slideData, preset) {
  const title = String(slideData.title || '').trim();
  const subtitle = String(slideData.subtitle || '').trim();
  const kicker = String(slideData.kicker || '').trim();
  const footer = String(slideData.footer || '').trim();
  const notes = String(slideData.notes || '').trim();

  slide.background = { color: 'FFFFFF' };

  if (kicker) {
    slide.addText(kicker, {
      x: 0.60,
      y: 0.42,
      w: 3.5,
      h: 0.22,
      margin: 0,
      fontFace: preset.font_body,
      fontSize: 8.5,
      color: '4B5563',
      bold: false,
      breakLine: false,
      valign: 'middle',
    });
  }

  slide.addText(title, {
    x: 0.60,
    y: 1.05,
    w: 8.80,
    h: 1.25,
    margin: 0,
    fontFace: preset.font_heading,
    fontSize: 30,
    color: '0B2545',
    bold: true,
    breakLine: false,
    valign: 'top',
    fit: 'shrink',
  });

  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.60,
      y: 2.70,
      w: 8.60,
      h: 1.00,
      margin: 0,
      fontFace: preset.font_body,
      fontSize: 13.5,
      color: '4B5563',
      bold: false,
      breakLine: false,
      valign: 'top',
      fit: 'shrink',
    });
  }

  if (footer) {
    slide.addText(footer, {
      x: 0.60,
      y: 5.05,
      w: 8.80,
      h: 0.18,
      margin: 0,
      fontFace: preset.font_body,
      fontSize: 7.5,
      color: '6B7280',
      bold: false,
      breakLine: false,
      valign: 'middle',
    });
  }

  if (notes) {
    slide.addNotes(notes);
  }
};

require(builderPath);
