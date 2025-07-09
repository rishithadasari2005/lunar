"""
Photometric Models for Lunar Surface Analysis

This module implements various photometric models used in photoclinometry
for lunar surface analysis, including the Hapke model, Lommel-Seeliger model,
and Lambertian model.
"""

import numpy as np
from abc import ABC, abstractmethod


class PhotometricModel(ABC):
    """
    Abstract base class for photometric models.
    
    All photometric models should inherit from this class and implement
    the compute_reflectance method.
    """
    
    @abstractmethod
    def compute_reflectance(self, normal, solar_dir, view_dir):
        """
        Compute surface reflectance given surface normal and illumination/viewing directions.
        
        Args:
            normal (numpy.ndarray): Surface normal vector [nx, ny, nz]
            solar_dir (numpy.ndarray): Solar illumination direction [sx, sy, sz]
            view_dir (numpy.ndarray): Viewing direction [vx, vy, vz]
            
        Returns:
            float: Computed reflectance value
        """
        pass
    
    def _compute_angles(self, normal, solar_dir, view_dir):
        """
        Compute geometric angles for photometric calculations.
        
        Args:
            normal (numpy.ndarray): Surface normal vector
            solar_dir (numpy.ndarray): Solar direction vector
            view_dir (numpy.ndarray): Viewing direction vector
            
        Returns:
            tuple: (cos_i, cos_e, cos_g) cosine of incidence, emission, and phase angles
        """
        # Ensure vectors are normalized
        normal = normal / np.linalg.norm(normal)
        solar_dir = solar_dir / np.linalg.norm(solar_dir)
        view_dir = view_dir / np.linalg.norm(view_dir)
        
        # Compute cosine of incidence angle (angle between normal and solar direction)
        cos_i = np.dot(normal, solar_dir)
        cos_i = np.clip(cos_i, 0, 1)  # Clamp to [0, 1]
        
        # Compute cosine of emission angle (angle between normal and viewing direction)
        cos_e = np.dot(normal, view_dir)
        cos_e = np.clip(cos_e, 0, 1)  # Clamp to [0, 1]
        
        # Compute cosine of phase angle (angle between solar and viewing directions)
        cos_g = np.dot(solar_dir, view_dir)
        cos_g = np.clip(cos_g, -1, 1)  # Clamp to [-1, 1]
        
        return cos_i, cos_e, cos_g


class LambertianModel(PhotometricModel):
    """
    Lambertian (cosine) photometric model.
    
    The simplest photometric model that assumes uniform scattering in all directions.
    Good for rough, matte surfaces but less accurate for lunar regolith.
    """
    
    def __init__(self, albedo=0.1):
        """
        Initialize Lambertian model.
        
        Args:
            albedo (float): Surface albedo (reflectance at normal incidence)
        """
        self.albedo = albedo
    
    def compute_reflectance(self, normal, solar_dir, view_dir):
        """
        Compute Lambertian reflectance.
        
        Args:
            normal (numpy.ndarray): Surface normal vector
            solar_dir (numpy.ndarray): Solar direction vector
            view_dir (numpy.ndarray): Viewing direction vector
            
        Returns:
            float: Lambertian reflectance
        """
        cos_i, cos_e, cos_g = self._compute_angles(normal, solar_dir, view_dir)
        
        # Lambertian model: R = (albedo / pi) * cos_i
        # Note: cos_e is not included as Lambertian scattering is uniform
        reflectance = (self.albedo / np.pi) * cos_i
        
        return reflectance


class LommelSeeligerModel(PhotometricModel):
    """
    Lommel-Seeliger photometric model.
    
    A more sophisticated model that accounts for shadowing and multiple scattering.
    Better approximation for lunar surfaces than Lambertian model.
    """
    
    def __init__(self, albedo=0.1):
        """
        Initialize Lommel-Seeliger model.
        
        Args:
            albedo (float): Surface albedo
        """
        self.albedo = albedo
    
    def compute_reflectance(self, normal, solar_dir, view_dir):
        """
        Compute Lommel-Seeliger reflectance.
        
        Args:
            normal (numpy.ndarray): Surface normal vector
            solar_dir (numpy.ndarray): Solar direction vector
            view_dir (numpy.ndarray): Viewing direction vector
            
        Returns:
            float: Lommel-Seeliger reflectance
        """
        cos_i, cos_e, cos_g = self._compute_angles(normal, solar_dir, view_dir)
        
        # Lommel-Seeliger model: R = (albedo / 4) * cos_i / (cos_i + cos_e)
        # This accounts for shadowing effects
        if cos_i + cos_e > 0:
            reflectance = (self.albedo / 4.0) * cos_i / (cos_i + cos_e)
        else:
            reflectance = 0.0
            
        return reflectance


class HapkeModel(PhotometricModel):
    """
    Hapke photometric model.
    
    The most sophisticated model for lunar surfaces, accounting for:
    - Single particle scattering
    - Multiple scattering
    - Surface roughness
    - Opposition effect
    
    Based on Hapke (1981) theory.
    """
    
    def __init__(self, w=0.1, b=0.5, c=0.1, h=0.1, theta=0.0, B0=1.0, h0=0.1):
        """
        Initialize Hapke model.
        
        Args:
            w (float): Single scattering albedo
            b (float): Backscatter parameter (0-1)
            c (float): Forward scatter parameter (0-1)
            h (float): Opposition effect parameter
            theta (float): Surface roughness angle (radians)
            B0 (float): Opposition effect strength
            h0 (float): Opposition effect angular width
        """
        self.w = w
        self.b = b
        self.c = c
        self.h = h
        self.theta = theta
        self.B0 = B0
        self.h0 = h0
    
    def compute_reflectance(self, normal, solar_dir, view_dir):
        """
        Compute Hapke reflectance.
        
        Args:
            normal (numpy.ndarray): Surface normal vector
            solar_dir (numpy.ndarray): Solar direction vector
            view_dir (numpy.ndarray): Viewing direction vector
            
        Returns:
            float: Hapke reflectance
        """
        cos_i, cos_e, cos_g = self._compute_angles(normal, solar_dir, view_dir)
        
        # Phase angle
        g = np.arccos(cos_g)
        
        # Single particle scattering function
        P = self._single_scattering_function(g)
        
        # Multiple scattering function
        H = self._multiple_scattering_function(cos_i, cos_e)
        
        # Opposition effect
        B = self._opposition_effect(g)
        
        # Surface roughness correction
        S = self._surface_roughness_correction(cos_i, cos_e, g)
        
        # Hapke model: R = (w / 4) * (cos_i / (cos_i + cos_e)) * P * H * (1 + B) * S
        if cos_i + cos_e > 0:
            reflectance = (self.w / 4.0) * (cos_i / (cos_i + cos_e)) * P * H * (1 + B) * S
        else:
            reflectance = 0.0
            
        return reflectance
    
    def _single_scattering_function(self, g):
        """
        Compute single particle scattering function.
        
        Args:
            g (float): Phase angle (radians)
            
        Returns:
            float: Single scattering function value
        """
        # Henyey-Greenstein function
        cos_g = np.cos(g)
        
        # Asymmetry parameter
        g_asym = self.c - self.b
        
        # Single scattering function
        P = (1 - g_asym**2) / (1 + g_asym**2 + 2 * g_asym * cos_g)**1.5
        
        return P
    
    def _multiple_scattering_function(self, cos_i, cos_e):
        """
        Compute multiple scattering function.
        
        Args:
            cos_i (float): Cosine of incidence angle
            cos_e (float): Cosine of emission angle
            
        Returns:
            float: Multiple scattering function value
        """
        # Approximate H function using Hapke's approximation
        gamma = np.sqrt(1 - self.w)
        H_i = (1 + 2 * cos_i) / (1 + 2 * gamma * cos_i)
        H_e = (1 + 2 * cos_e) / (1 + 2 * gamma * cos_e)
        
        return H_i * H_e
    
    def _opposition_effect(self, g):
        """
        Compute opposition effect.
        
        Args:
            g (float): Phase angle (radians)
            
        Returns:
            float: Opposition effect value
        """
        # Opposition effect using exponential function
        B = self.B0 / (1 + np.tan(g / 2) / self.h0)
        
        return B
    
    def _surface_roughness_correction(self, cos_i, cos_e, g):
        """
        Compute surface roughness correction.
        
        Args:
            cos_i (float): Cosine of incidence angle
            cos_e (float): Cosine of emission angle
            g (float): Phase angle (radians)
            
        Returns:
            float: Surface roughness correction factor
        """
        if self.theta == 0:
            return 1.0
        
        # Simplified roughness correction
        # For small roughness angles, this is approximately 1
        S = 1.0 - 0.1 * self.theta * (1 - cos_i * cos_e)
        
        return np.clip(S, 0.5, 1.5)


class LunarPhotometricModel(PhotometricModel):
    """
    Specialized lunar photometric model combining multiple effects.
    
    This model is specifically tuned for lunar surface characteristics
    and includes empirical corrections based on lunar observations.
    """
    
    def __init__(self, albedo=0.12, phase_function="hapke", **kwargs):
        """
        Initialize lunar photometric model.
        
        Args:
            albedo (float): Lunar surface albedo (typically 0.07-0.15)
            phase_function (str): Phase function type ('hapke', 'lommel_seeliger')
            **kwargs: Additional parameters
        """
        self.albedo = albedo
        
        if phase_function.lower() == "hapke":
            self.phase_model = HapkeModel(**kwargs)
        elif phase_function.lower() == "lommel_seeliger":
            self.phase_model = LommelSeeligerModel(albedo=albedo)
        else:
            raise ValueError(f"Unknown phase function: {phase_function}")
    
    def compute_reflectance(self, normal, solar_dir, view_dir):
        """
        Compute lunar surface reflectance.
        
        Args:
            normal (numpy.ndarray): Surface normal vector
            solar_dir (numpy.ndarray): Solar direction vector
            view_dir (numpy.ndarray): Viewing direction vector
            
        Returns:
            float: Lunar surface reflectance
        """
        # Get base reflectance from phase model
        base_reflectance = self.phase_model.compute_reflectance(normal, solar_dir, view_dir)
        
        # Apply lunar-specific corrections
        corrected_reflectance = self._apply_lunar_corrections(
            base_reflectance, normal, solar_dir, view_dir
        )
        
        return corrected_reflectance
    
    def _apply_lunar_corrections(self, base_reflectance, normal, solar_dir, view_dir):
        """
        Apply lunar-specific corrections to base reflectance.
        
        Args:
            base_reflectance (float): Base reflectance from phase model
            normal (numpy.ndarray): Surface normal vector
            solar_dir (numpy.ndarray): Solar direction vector
            view_dir (numpy.ndarray): Viewing direction vector
            
        Returns:
            float: Corrected reflectance
        """
        cos_i, cos_e, cos_g = self._compute_angles(normal, solar_dir, view_dir)
        
        # Lunar surface roughness correction (empirical)
        roughness_factor = 1.0 + 0.1 * (1 - cos_i * cos_e)
        
        # Lunar phase curve correction (empirical)
        g = np.arccos(cos_g)
        phase_correction = 1.0 + 0.05 * np.exp(-g / np.radians(20))
        
        # Apply corrections
        corrected_reflectance = base_reflectance * roughness_factor * phase_correction
        
        return corrected_reflectance 