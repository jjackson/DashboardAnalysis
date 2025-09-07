#!/usr/bin/env python3
"""
Pydantic models for CommCare form data validation and flattening.
Handles schema evolution across different app versions gracefully.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
import json

class ChildInfo(BaseModel):
    """Child demographic information."""
    child_name: Optional[str] = None
    child_full_name: Optional[str] = None
    child_first_name: Optional[str] = None
    childs_last_name: Optional[str] = None
    child_middle_name: Optional[str] = None
    childs_dob: Optional[str] = None
    childs_gender: Optional[str] = None
    childs_age_in_months: Optional[int] = None
    childs_age_in_years: Optional[int] = None
    childs_age_in_weeks: Optional[int] = None
    child_unique_id: Optional[str] = None

class HealthMetrics(BaseModel):
    """Health screening and measurement data."""
    muac_colour: Optional[str] = None
    soliciter_muac_cm: Optional[float] = None
    muac_malnutrition: Optional[str] = None
    diagnosed_with_mal_past_3_months: Optional[str] = None
    under_treatment_for_mal: Optional[str] = None
    malnutrition_screening_status: Optional[str] = None

class VaccinationInfo(BaseModel):
    """Vaccination and vitamin A information."""
    received_any_vaccine: Optional[str] = None
    vaccines_received: Optional[str] = None
    received_va_dose_before: Optional[str] = None
    recent_va_dose: Optional[str] = None
    va_child_unwell_today: Optional[str] = None
    have_glasses: Optional[str] = None

class HouseholdInfo(BaseModel):
    """Household and family information."""
    fetch_hh_head_name: Optional[str] = None
    fetch_hh_phone_number: Optional[str] = None
    fetch_hh_village_name: Optional[str] = None
    fetch_hh_have_children: Optional[str] = None
    fetch_hh_living_children_of_age: Optional[str] = None
    load_household_unique_id: Optional[str] = None
    load_household_full_name: Optional[str] = None
    no_of_children: Optional[int] = None

class LocationInfo(BaseModel):
    """GPS and location data."""
    gps_location_child_registration: Optional[str] = None
    normalized_location_cr: Optional[str] = None
    fetch_hh_gps_location: Optional[str] = None

class MetadataInfo(BaseModel):
    """Form submission metadata."""
    username: Optional[str] = None
    deviceID: Optional[str] = None
    timeStart: Optional[str] = None
    timeEnd: Optional[str] = None
    appVersion: Optional[str] = None
    commcare_version: Optional[str] = None
    app_build_version: Optional[int] = None
    instanceID: Optional[str] = None

class FlattenedFormData(BaseModel):
    """Flattened and validated form data structure."""
    
    # Core identifiers
    opportunity_id: int
    flw_id: int
    flw_name: Optional[str] = None
    visit_date: Optional[datetime] = None
    
    # Child information
    child: ChildInfo = Field(default_factory=ChildInfo)
    
    # Health metrics
    health: HealthMetrics = Field(default_factory=HealthMetrics)
    
    # Vaccination info
    vaccination: VaccinationInfo = Field(default_factory=VaccinationInfo)
    
    # Household info
    household: HouseholdInfo = Field(default_factory=HouseholdInfo)
    
    # Location info
    location: LocationInfo = Field(default_factory=LocationInfo)
    
    # Metadata
    metadata: MetadataInfo = Field(default_factory=MetadataInfo)
    
    # Raw form JSON for debugging
    raw_form_json: Optional[str] = None
    
    @classmethod
    def from_raw_record(cls, record: Dict[str, Any]) -> 'FlattenedFormData':
        """
        Create a FlattenedFormData instance from a raw CSV record.
        Handles JSON parsing and graceful field extraction.
        """
        try:
            # Parse the form_json string (handle both JSON and Python dict formats)
            if isinstance(record.get('form_json'), str):
                json_str = record['form_json']
                try:
                    # Try JSON first
                    form_data = json.loads(json_str)
                except json.JSONDecodeError:
                    # Try Python literal_eval for dict format
                    import ast
                    form_data = ast.literal_eval(json_str)
            else:
                form_data = record.get('form_json', {})
        except (json.JSONDecodeError, TypeError, ValueError, SyntaxError):
            form_data = {}
        
        # Extract nested data safely
        form_dict = form_data.get('form', {}) if isinstance(form_data, dict) else {}
        metadata_dict = form_data.get('metadata', {}) if isinstance(form_data, dict) else {}
        
        # Create child info
        child_info = ChildInfo(
            child_name=form_dict.get('child_name'),
            child_full_name=form_dict.get('child_full_name'),
            child_first_name=form_dict.get('child_first_name'),
            childs_last_name=form_dict.get('childs_last_name'),
            child_middle_name=form_dict.get('child_middle_name'),
            childs_dob=form_dict.get('childs_dob'),
            childs_gender=form_dict.get('childs_gender'),
            childs_age_in_months=cls._safe_int(form_dict.get('childs_age_in_months')),
            childs_age_in_years=cls._safe_int(form_dict.get('childs_age_in_years')),
            childs_age_in_weeks=cls._safe_int(form_dict.get('childs_age_in_weeks')),
            child_unique_id=form_dict.get('child_unique_id')
        )
        
        # Create health metrics (extract from nested structures - handle both fake and real data formats)
        health_info = cls._extract_health_metrics(form_dict)
        
        # Create vaccination info
        vaccination_info = VaccinationInfo(
            received_any_vaccine=form_dict.get('received_any_vaccine'),
            vaccines_received=form_dict.get('vaccines_received'),
            received_va_dose_before=form_dict.get('received_va_dose_before'),
            recent_va_dose=form_dict.get('recent_va_dose'),
            va_child_unwell_today=form_dict.get('va_child_unwell_today'),
            have_glasses=form_dict.get('have_glasses')
        )
        
        # Create household info
        household_info = HouseholdInfo(
            fetch_hh_head_name=form_dict.get('fetch_hh_head_name'),
            fetch_hh_phone_number=form_dict.get('fetch_hh_phone_number'),
            fetch_hh_village_name=form_dict.get('fetch_hh_village_name'),
            fetch_hh_have_children=form_dict.get('fetch_hh_have_children'),
            fetch_hh_living_children_of_age=form_dict.get('fetch_hh_living_children_of_age'),
            load_household_unique_id=form_dict.get('load_household_unique_id'),
            load_household_full_name=form_dict.get('load_household_full_name'),
            no_of_children=cls._safe_int(form_dict.get('no_of_children'))
        )
        
        # Create location info
        location_info = LocationInfo(
            gps_location_child_registration=form_dict.get('gps_location_child_registration'),
            normalized_location_cr=form_dict.get('normalized_location_cr'),
            fetch_hh_gps_location=form_dict.get('fetch_hh_gps_location')
        )
        
        # Create metadata info
        metadata_info = MetadataInfo(
            username=metadata_dict.get('username'),
            deviceID=metadata_dict.get('deviceID'),
            timeStart=metadata_dict.get('timeStart'),
            timeEnd=metadata_dict.get('timeEnd'),
            appVersion=metadata_dict.get('appVersion'),
            commcare_version=metadata_dict.get('commcare_version'),
            app_build_version=cls._safe_int(metadata_dict.get('app_build_version')),
            instanceID=metadata_dict.get('instanceID')
        )
        
        # Extract visit_date from form_json if not in record
        visit_date = record.get('visit_date')
        if visit_date is None and isinstance(form_data, dict):
            # Try to get from metadata
            visit_date = metadata_dict.get('timeStart') or metadata_dict.get('timeEnd')
            if visit_date:
                try:
                    visit_date = pd.to_datetime(visit_date)
                except:
                    visit_date = None
        
        return cls(
            opportunity_id=record['opportunity_id'],
            flw_id=record['flw_id'],
            flw_name=record.get('flw_name'),
            visit_date=visit_date,
            child=child_info,
            health=health_info,
            vaccination=vaccination_info,
            household=household_info,
            location=location_info,
            metadata=metadata_info,
            raw_form_json=record.get('form_json')
        )
    
    @staticmethod
    def _safe_int(value) -> Optional[int]:
        """Safely convert value to int, return None if invalid."""
        if value is None or value == '':
            return None
        try:
            return int(float(str(value)))
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float, return None if invalid."""
        if value is None or value == '':
            return None
        try:
            return float(str(value))
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _extract_health_metrics(cls, form_dict: Dict[str, Any]) -> 'HealthMetrics':
        """
        Extract health metrics from form data, handling different JSON structures.
        
        Fake data structure:
        - muac_colour: direct field
        - muac_display_group_1.soliciter_muac_cm: nested
        
        Real data structure:
        - muac_group.Muac.vitals.muac_colour: deeply nested
        - muac_group.Muac.vitals.soliciter_muac_cm: deeply nested
        """
        
        # Initialize default values
        muac_colour = None
        soliciter_muac_cm = None
        muac_malnutrition = None
        diagnosed_with_mal_past_3_months = None
        under_treatment_for_mal = None
        malnutrition_screening_status = None
        
        # Try fake data structure first (muac_display_group_1)
        if 'muac_display_group_1' in form_dict:
            muac_group = form_dict['muac_display_group_1']
            muac_colour = form_dict.get('muac_colour')
            soliciter_muac_cm = cls._safe_float(muac_group.get('soliciter_muac_cm'))
            muac_malnutrition = form_dict.get('muac_malnutrition')
            diagnosed_with_mal_past_3_months = muac_group.get('diagnosed_with_mal_past_3_months')
            under_treatment_for_mal = muac_group.get('under_treatment_for_mal')
            malnutrition_screening_status = form_dict.get('malnutrition_screening_status')
        
        # Try real data structure (muac_group.Muac.vitals)
        elif 'muac_group' in form_dict:
            muac_group = form_dict.get('muac_group', {})
            if isinstance(muac_group, dict) and 'Muac' in muac_group:
                muac_data = muac_group['Muac']
                if isinstance(muac_data, dict) and 'vitals' in muac_data:
                    vitals = muac_data['vitals']
                    if isinstance(vitals, dict):
                        muac_colour = vitals.get('muac_colour')
                        soliciter_muac_cm = cls._safe_float(vitals.get('soliciter_muac_cm'))
                        muac_malnutrition = vitals.get('muac_malnutrition')
                        diagnosed_with_mal_past_3_months = vitals.get('diagnosed_with_mal_past_3_months')
                        under_treatment_for_mal = vitals.get('under_treatment_for_mal')
        
        # Fallback: try direct fields (in case of other structures)
        if muac_colour is None:
            muac_colour = form_dict.get('muac_colour')
        if soliciter_muac_cm is None:
            soliciter_muac_cm = cls._safe_float(form_dict.get('soliciter_muac_cm'))
        if muac_malnutrition is None:
            muac_malnutrition = form_dict.get('muac_malnutrition')
        if malnutrition_screening_status is None:
            malnutrition_screening_status = form_dict.get('malnutrition_screening_status')
        
        return HealthMetrics(
            muac_colour=muac_colour,
            soliciter_muac_cm=soliciter_muac_cm,
            muac_malnutrition=muac_malnutrition,
            diagnosed_with_mal_past_3_months=diagnosed_with_mal_past_3_months,
            under_treatment_for_mal=under_treatment_for_mal,
            malnutrition_screening_status=malnutrition_screening_status
        )
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for DataFrame creation."""
        flat_dict = {
            # Core fields
            'opportunity_id': self.opportunity_id,
            'flw_id': self.flw_id,
            'flw_name': self.flw_name,
            'visit_date': self.visit_date,
            
            # Child info (flattened with prefix)
            'child_name': self.child.child_name,
            'child_full_name': self.child.child_full_name,
            'child_first_name': self.child.child_first_name,
            'childs_last_name': self.child.childs_last_name,
            'child_middle_name': self.child.child_middle_name,
            'childs_dob': self.child.childs_dob,
            'childs_gender': self.child.childs_gender,
            'childs_age_in_months': self.child.childs_age_in_months,
            'childs_age_in_years': self.child.childs_age_in_years,
            'childs_age_in_weeks': self.child.childs_age_in_weeks,
            'child_unique_id': self.child.child_unique_id,
            
            # Health metrics
            'muac_colour': self.health.muac_colour,
            'soliciter_muac_cm': self.health.soliciter_muac_cm,
            'muac_malnutrition': self.health.muac_malnutrition,
            'diagnosed_with_mal_past_3_months': self.health.diagnosed_with_mal_past_3_months,
            'under_treatment_for_mal': self.health.under_treatment_for_mal,
            'malnutrition_screening_status': self.health.malnutrition_screening_status,
            
            # Vaccination info
            'received_any_vaccine': self.vaccination.received_any_vaccine,
            'vaccines_received': self.vaccination.vaccines_received,
            'received_va_dose_before': self.vaccination.received_va_dose_before,
            'recent_va_dose': self.vaccination.recent_va_dose,
            'va_child_unwell_today': self.vaccination.va_child_unwell_today,
            'have_glasses': self.vaccination.have_glasses,
            
            # Household info
            'fetch_hh_head_name': self.household.fetch_hh_head_name,
            'fetch_hh_phone_number': self.household.fetch_hh_phone_number,
            'fetch_hh_village_name': self.household.fetch_hh_village_name,
            'fetch_hh_have_children': self.household.fetch_hh_have_children,
            'fetch_hh_living_children_of_age': self.household.fetch_hh_living_children_of_age,
            'load_household_unique_id': self.household.load_household_unique_id,
            'load_household_full_name': self.household.load_household_full_name,
            'no_of_children': self.household.no_of_children,
            
            # Location info
            'gps_location_child_registration': self.location.gps_location_child_registration,
            'normalized_location_cr': self.location.normalized_location_cr,
            'fetch_hh_gps_location': self.location.fetch_hh_gps_location,
            
            # Metadata
            'username': self.metadata.username,
            'deviceID': self.metadata.deviceID,
            'timeStart': self.metadata.timeStart,
            'timeEnd': self.metadata.timeEnd,
            'appVersion': self.metadata.appVersion,
            'commcare_version': self.metadata.commcare_version,
            'app_build_version': self.metadata.app_build_version,
            'instanceID': self.metadata.instanceID
        }
        
        return flat_dict
