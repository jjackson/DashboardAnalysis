#!/usr/bin/env python3
"""
Streamlined form model for fake detection analysis.
Focuses only on the 18 successfully extracted team-recommended fields
plus core metadata (opportunity_id, flw_id, visit_date).
"""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import json
import ast

class CoreMetadata(BaseModel):
    """Core metadata fields for analysis."""
    opportunity_id: Optional[int] = None
    flw_id: Optional[int] = None
    visit_date: Optional[str] = None

class TargetedFormData(BaseModel):
    """
    Streamlined form data model with only the 18 successfully extracted fields
    plus core metadata for fake detection analysis.
    """
    
    # Core metadata
    core: CoreMetadata
    
    # Demographics & Household (7 fields)
    childs_age_in_month: Optional[int] = None
    childs_gender: Optional[str] = None
    child_name: Optional[str] = None
    no_of_children: Optional[int] = None
    hh_have_children: Optional[str] = None
    household_phone: Optional[str] = None
    household_name: Optional[str] = None
    
    # Health & MUAC (4 fields)
    diagnosed_with_mal_past_3_months: Optional[str] = None
    muac_colour: Optional[str] = None
    under_treatment_for_mal: Optional[str] = None
    soliciter_muac_cm: Optional[float] = None
    
    # Vaccination & Health (5 fields)
    have_glasses: Optional[str] = None
    received_va_dose_before: Optional[str] = None
    received_any_vaccine: Optional[str] = None
    va_child_unwell_today: Optional[str] = None
    recent_va_dose: Optional[str] = None
    
    # Recovery (2 fields)
    diarrhea_last_month: Optional[str] = None
    did_the_child_recover: Optional[str] = None
    
    @classmethod
    def from_raw_record(cls, record: Dict[str, Any]) -> 'TargetedFormData':
        """
        Create a TargetedFormData instance from a raw CSV record.
        Uses the discovered field paths from the targeted field extractor.
        """
        try:
            # Parse the form_json string
            if isinstance(record.get('form_json'), str):
                json_str = record['form_json']
                try:
                    form_data = json.loads(json_str)
                except json.JSONDecodeError:
                    form_data = ast.literal_eval(json_str)
            else:
                form_data = record.get('form_json', {})
        except (json.JSONDecodeError, TypeError, ValueError, SyntaxError):
            form_data = {}
        
        # Extract core metadata
        core_metadata = CoreMetadata(
            opportunity_id=cls._safe_int(record.get('opportunity_id')),
            flw_id=cls._safe_int(record.get('flw_id')),
            visit_date=cls._extract_visit_date(form_data, record)
        )
        
        # Extract all target fields using multiple path strategies
        extracted_fields = {}
        
        # Define field extraction paths (from our successful discovery)
        field_paths = {
            # Demographics & Household
            'childs_age_in_month': [
                'form.additional_case_info.childs_age_in_month',
                'form.case.update.childs_age_in_months',
                'form.childs_age_in_months'
            ],
            'childs_gender': [
                'form.additional_case_info.childs_gender',
                'form.childs_gender'
            ],
            'child_name': [
                'form.additional_case_info.child_name',
                'form.case.update.child_name'
            ],
            'no_of_children': [
                'form.additional_case_info.no_of_children',
                'form.subcase_0.case.update.fetch_hh_living_children_of_age',
                'form.fetch_hh_living_children_of_age'
            ],
            'hh_have_children': [
                'form.additional_case_info.hh_have_children',
                'form.subcase_0.case.update.fetch_hh_have_children',
                'form.fetch_hh_have_children'
            ],
            'household_phone': [
                'form.additional_case_info.household_phone',
                'form.subcase_0.case.update.fetch_hh_phone_number',
                'form.fetch_hh_phone_number'
            ],
            'household_name': [
                'form.additional_case_info.household_name',
                'form.load_household_full_name',
                'form.subcase_0.case.update.fetch_hh_head_name',
                'form.fetch_hh_head_name'
            ],
            
            # Health & MUAC
            'diagnosed_with_mal_past_3_months': [
                'form.case.update.diagnosed_with_mal_past_3_months',
                'form.subcase_0.case.update.diagnosed_with_mal_past_3_months',
                'form.muac_group.muac_display_group_1.diagnosed_with_mal_past_3_months',
                'form.muac_group.Muac.vitals.diagnosed_with_mal_past_3_months'
            ],
            'muac_colour': [
                'form.case.update.muac_colour',
                'form.subcase_0.case.update.muac_colour',
                'form.muac_group.muac_colour',
                'form.muac_group.Muac.vitals.muac_colour'
            ],
            'under_treatment_for_mal': [
                'form.case.update.under_treatment_for_mal',
                'form.subcase_0.case.update.under_treatment_for_mal',
                'form.muac_group.muac_display_group_1.under_treatment_for_mal',
                'form.muac_group.Muac.vitals.under_treatment_for_mal'
            ],
            'soliciter_muac_cm': [
                'form.case.update.soliciter_muac_cm',
                'form.subcase_0.case.update.soliciter_muac_cm',
                'form.muac_group.muac_display_group_1.soliciter_muac_cm',
                'form.muac_group.Muac.vitals.soliciter_muac_cm'
            ],
            
            # Vaccination & Health
            'have_glasses': [
                'form.have_glasses',
                'form.case.update.have_glasses',
                'form.vita_group.have_glasses',
                'form.va.have_glasses'
            ],
            'received_va_dose_before': [
                'form.received_va_dose_before',
                'form.case.update.received_va_dose_before',
                'form.vita_group.received_va_dose_before',
                'form.va.received_va_dose_before'
            ],
            'received_any_vaccine': [
                'form.received_any_vaccine',
                'form.case.update.received_any_vaccine',
                'form.vita_group.pictures.received_any_vaccine',
                'form.pictures.received_any_vaccine'
            ],
            'va_child_unwell_today': [
                'form.va_child_unwell_today',
                'form.case.update.va_child_unwell_today',
                'form.vita_group.va_child_unwell_today',
                'form.va.va_child_unwell_today'
            ],
            'recent_va_dose': [
                'form.recent_va_dose',
                'form.case.update.recent_va_dose',
                'form.vita_group.recent_va_dose'
            ],
            
            # Recovery
            'diarrhea_last_month': [
                'form.diarrhea_last_month',
                'form.case.update.diarrhea_last_month',
                'form.ors_group.diarrhea_last_month',
                'form.ors.diarrhea_last_month'
            ],
            'did_the_child_recover': [
                'form.did_the_child_recover',
                'form.case.update.did_the_child_recover',
                'form.ors_group.did_the_child_recover',
                'form.ors.did_the_child_recover'
            ]
        }
        
        # Extract each field using multiple path attempts
        for field_name, paths in field_paths.items():
            extracted_fields[field_name] = cls._extract_field_with_paths(form_data, paths)
        
        # Apply type conversions
        extracted_fields['childs_age_in_month'] = cls._safe_int(extracted_fields['childs_age_in_month'])
        extracted_fields['no_of_children'] = cls._safe_int(extracted_fields['no_of_children'])
        extracted_fields['soliciter_muac_cm'] = cls._safe_float(extracted_fields['soliciter_muac_cm'])
        
        return cls(
            core=core_metadata,
            **extracted_fields
        )
    
    @staticmethod
    def _extract_field_with_paths(form_data: Dict, paths: list) -> Any:
        """Extract field value trying multiple paths in order."""
        for path_str in paths:
            try:
                path_parts = path_str.split('.')
                current = form_data
                
                for step in path_parts:
                    if step == '[0]':
                        current = current[0] if isinstance(current, list) and current else None
                    else:
                        current = current.get(step) if isinstance(current, dict) else None
                    
                    if current is None:
                        break
                
                if current is not None:
                    return current
            except (KeyError, TypeError, IndexError):
                continue
        
        return None
    
    @staticmethod
    def _extract_visit_date(form_data: Dict, record: Dict) -> Optional[str]:
        """Extract visit date from various possible locations."""
        # Try direct field first
        if 'visit_date' in record:
            return record['visit_date']
        
        # Try metadata locations
        metadata_paths = [
            'metadata.timeStart',
            'metadata.timeEnd',
            'form.meta.timeStart',
            'form.meta.timeEnd'
        ]
        
        for path_str in metadata_paths:
            try:
                path_parts = path_str.split('.')
                current = form_data
                
                for step in path_parts:
                    current = current.get(step) if isinstance(current, dict) else None
                    if current is None:
                        break
                
                if current is not None:
                    # Convert timestamp to date if needed
                    if isinstance(current, str) and 'T' in current:
                        return current.split('T')[0]
                    return str(current)
            except (KeyError, TypeError):
                continue
        
        return None
    
    @staticmethod
    def _safe_int(value) -> Optional[int]:
        """Safely convert value to int."""
        if value is None or value == '':
            return None
        try:
            return int(float(str(value)))
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or value == '':
            return None
        try:
            return float(str(value))
        except (ValueError, TypeError):
            return None
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for DataFrame creation."""
        flat_dict = {
            # Core metadata
            'opportunity_id': self.core.opportunity_id,
            'flw_id': self.core.flw_id,
            'visit_date': self.core.visit_date,
            
            # Demographics & Household
            'childs_age_in_month': self.childs_age_in_month,
            'childs_gender': self.childs_gender,
            'child_name': self.child_name,
            'no_of_children': self.no_of_children,
            'hh_have_children': self.hh_have_children,
            'household_phone': self.household_phone,
            'household_name': self.household_name,
            
            # Health & MUAC
            'diagnosed_with_mal_past_3_months': self.diagnosed_with_mal_past_3_months,
            'muac_colour': self.muac_colour,
            'under_treatment_for_mal': self.under_treatment_for_mal,
            'soliciter_muac_cm': self.soliciter_muac_cm,
            
            # Vaccination & Health
            'have_glasses': self.have_glasses,
            'received_va_dose_before': self.received_va_dose_before,
            'received_any_vaccine': self.received_any_vaccine,
            'va_child_unwell_today': self.va_child_unwell_today,
            'recent_va_dose': self.recent_va_dose,
            
            # Recovery
            'diarrhea_last_month': self.diarrhea_last_month,
            'did_the_child_recover': self.did_the_child_recover
        }
        
        return flat_dict
